import os
import random
from math import radians, cos, sin, atan2, sqrt
from datetime import datetime
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import *
import seaborn as sns
import numpy as np
from collections import namedtuple

# For handling xy coordinates
POINT = namedtuple('POINT', 'point_id, x, y, freq')


class Trip:
    """
    Encapasulates a single trip
    """

    def __init__(self, id=None, start_time=None, origin=None, destination=None, end_time=None):
        """

        :param id:
        :param start_time:
        :param origin:
        :param destination:
        :param end_time:
        """
        self.trp_start_time = start_time
        self.trp_origin = origin
        self.trp_dest = destination
        self.trp_end_time = end_time
        self.trp_distance = None
        self.trp_id = id
        self.trp_date = None
        self.trp_hr = None

    def set_trip_distance(self):
        """

        :return:
        """
        distance = calculate_distance(pt1=(self.trp_origin.y, self.trp_origin.x),
                                      pt2=(self.trp_dest.y, self.trp_dest.x))
        self.trp_distance = distance

    def set_trp_time_attributes(self):
        """
        Set base time attributes
        :return:
        """
        self.trp_date = self.trp_start_time.date()
        self.trp_hr = self.trp_start_time.date()


class Cluster:
    def __init__(self, method='sequential', clust_type=None, clust_id=None, x=None, y=None,
                 last_visit=None, centre_type=None, members=None):
        """
        :param method: Method generating cluster. Default is sequential (Hartigan Leader)
        :param clust_type: they may be several cluster types (e.g., for trips or determining home)
        :param clust_id: unique cluster ID
        :param radius: Radius used when creating this cluster
        :param last_visit: Time stamp of when this cluster was last visited
        :param x: cluster centre x-Longitude
        :param y: cluster centre y-Latitude
        :param centre_type: whether centre is actual value is centroid
        """
        self.method = method
        self.clust_type = clust_type
        self.clust_id = clust_id
        self.x = x
        self.y = y
        self.last_visit = last_visit
        self.first_visit = last_visit
        self.centre_type = centre_type
        self.site_id = None
        self.adm1_id = None
        self.adm1_name = None
        self.adm2_id = None
        self.adm2_name = None
        self.adm3_name = None
        self.adm3_id = None
        self.grid_1km_id = None
        self.members = members
        self.stay_time = None
        self.stop_or_stay = None
        self.visit_freq = 0
        self.hr_visit_freq = {i: 0 for i in range(0, 24)}
        self.home_work_visit_freq = {'home': 0, 'work': 0, 'transit': 0}

    def __hash__(self):
        return hash(self.clust_id)

    def __eq__(self, other):
        if isinstance(other, Cluster):
            return self.clust_id == other.clust_id
        return NotImplemented

    def update_visit_times(self, time_col=None):
        """
        When a new member is added, update date of last visit
        :return:
        """
        members = self.members
        timestamps = [item[time_col] for item in members]
        self.last_visit = sorted(timestamps, reverse=True)[0]
        self.first_visit = sorted(timestamps, reverse=False)[0]

    def update_cluster_center(self):
        """
        Update cluster centre
        :return:
        """
        pts = [POINT(point_id=m['site_id'], x=m['x'], y=m['y'], freq=1) for m in self.members]
        new_clust_centre = find_geographic_centroid(locations_with_freqs=pts, weighted=False)
        self.x = new_clust_centre.x
        self.y = new_clust_centre.y

    def get_members_count(self):
        """
        Returns number of members in cluster
        :return:
        """
        return len(self.members)

    def stop_stay_categorisation(self, stay_threshold=15):
        """
        Since some places will be considered as trip stops rather than destinations or
        origins. Although this may have very little effect in CDR data.
        :param stay_threshold:  Minimum amount of time to qualify as stay
        :return:
        """
        if self.stay_time < stay_threshold:
            self.stop_or_stay = 'stop'
        else:
            self.stop_or_stay = 'OD'

    def update_hr_visit_freq(self, hr=None):
        """
        Updates visit frequencies
        :return:
        """
        hr_freqs = self.hr_visit_freq
        if hr in hr_freqs:
            hr_freqs[hr] += 1
            self.hr_visit_freq = hr_freqs
            return
        else:
            hr_freqs[hr] = 1
            self.hr_visit_freq = hr_freqs

    def update_visit_freq(self):
        """
        Updates total visit frequency. This is more for verification with counts from the hours.
        :return:
        """
        freq = self.visit_freq
        self.visit_freq = freq + 1

    def update_home_work_visit_freqs(self, hr=None, day=None, work_hrs=None, transit_hrs=None, exclude_days=None,
                                     home_hrs=None):
        """
        Updates visit frequencies
        :return:
        """
        if day in exclude_days:
            return

        freqs = self.home_work_visit_freq
        home = freqs['home']
        work = freqs['work']
        transit = freqs['transit']

        if hr in work_hrs:
            work += 1
            freqs['work'] = work
        if hr in home_hrs:
            home += 1
            freqs['home'] = home
        if hr in transit_hrs:
            transit += 1
            freqs['transit'] = transit


def find_geographic_centroid(locations_with_freqs=None, weighted=False):
    """
    Finds geographic centre, wither weighted by location visit frequency or not.
    :param radius: Radius (Km) for  for spanning locations
    :param locations_with_freqs: For finding weighted location
    :return:
    """

    lat = []
    lon = []
    sum_of_weights = 0
    for l in locations_with_freqs:
        if weighted:
            w = l.freq
        else:
            w = 1

        lat.append(l.y * w)
        lon.append(l.x * w)
        sum_of_weights += w

    y = sum(lat) / sum_of_weights
    x = sum(lon) / sum_of_weights
    pt = POINT(point_id='virtual_loc', x=x, y=y, freq=-1)
    return pt


def distance_matrix(xy_list=None):
    """
    Return distance matrix from a dictlist of xy coordinates
    :param xy_list:
    :return: a dataframe style of distance matrix
    """
    df = pd.DataFrame([dict(d._asdict()) for d in xy_list])

    for pt in xy_list:
        pt_id = pt.point_id
        colname = 'to_' + str(pt_id)
        df[colname] = df.apply(lambda x: calculate_distance(pt1=(x['x'], x['y']), pt2=(pt.x, pt.y)), axis=1)

    return df


def distances_travelled(unique_locs=None):
    """
    Daily distances travelled
    :return: a distance matrix containing distances travelled
    """
    df_distances = distance_matrix(xy_list=unique_locs)
    distance_cols = [c for c in df_distances.columns if "to" in c]
    df_distances_only = df_distances[distance_cols]
    dist_matrix = df_distances_only.values
    dist_matrix2 = dist_matrix[np.triu_indices(dist_matrix.shape[0], k=1)]

    return dist_matrix2


def chunks(n=None, lst=None):
    """
    Split list of clusters into trips
    :param n:
    :param lst:
    :return:
    """
    if len(lst) == 2:
        return [lst]

    n = min(n, len(lst) - 1)
    return [lst[i:i + n] for i in range(len(lst) - n + 1)]


def detect_trips(clusters=None):
    """
    Detect trips from location history
    :param clusters:
    :return:
    """
    # drop all stops
    [clusters.remove(c) for c in clusters if c.stop_or_stay == 'stop']

    # special cases
    if len(clusters) == 1:
        return None

    # generate trips
    clusters.sort(key=lambda x: x.clust_id, reverse=False)

    # split into pairs
    od_pairs = chunks(lst=clusters, n=2)

    # create trips and characterize them
    trips = []
    for i, od in enumerate(od_pairs):
        origin = od[0]
        dest = od[1]
        # just double check that start time is
        # assert origin.last_visit < dest.last_visit
        trp = Trip(id=i, origin=origin, destination=dest, start_time=origin.last_visit, end_time=dest.first_visit)
        trp.set_trip_distance()
        trp.set_trp_time_attributes()
        trips.append(trp)

    return trips


def generate_cluster_stay_time(clusters=None):
    """
    Estimates how long a user stayed at some cluster. The
    final cluster-last visited place is given a stay time of
    1 day.
    :param clusters: Clusters to evaluate
    :return:
    """
    res = []
    for i, x in enumerate(clusters):
        diff = (x.first_visit - clusters[i - 1].first_visit).total_seconds()
        stay_time = int(diff / 60)
        res.append(stay_time)

    for s, c in zip(res[1:], clusters):
        c.stay_time = s
    clusters[-1].stay_time = 1440

    # categorise as stay or stops
    [c.stop_stay_categorisation() for c in clusters]

    return clusters


def cluster_cells_within_radius_trps_version(radius=1, loc_history: pd.DataFrame = None, time_col=None, x=None, y=None):
    """
    Given location history in a time interval, cluster them.
    :param radius: Radius (Km) for  for spanning locations
    :param locations_with_freqs: For finding weighted location
    :return:
    """
    # sort the events by timestamp
    loc_history.sort_values(by=time_col, ascending=True, inplace=True)

    # initiate some cluster variables
    clusters = []

    # initiate clusters by assigning first element to first cluster
    first_loc = loc_history.iloc[0]
    first_clust = Cluster(clust_id=1, x=first_loc[x], y=first_loc[y], last_visit=first_loc[time_col],
                          members=[{'x': first_loc[x], 'y': first_loc[y], 'site_id': first_loc['site_id'],
                                    time_col: first_loc[time_col]}])
    first_clust.update_visit_freq()
    first_clust.update_hr_visit_freq(hr=first_loc[time_col].hour)
    clusters.append(first_clust)

    # loop through loc_history
    for i, row in loc_history[1:].iterrows():
        current_member = {'x': row[x], 'y': row[y], 'site_id': row['site_id'],
                          time_col: row[time_col]}

        # get most recent cluster
        recent_clust = clusters[-1]

        # distance between recent clust centre and current loc
        dist = calculate_distance(pt1=(recent_clust.y, recent_clust.x),
                                  pt2=(current_member['y'], current_member['x']))

        # add to existing cluster and update
        if dist <= radius:
            # add new member
            recent_clust.members.append(current_member)

            # update cluster centre
            recent_clust.update_cluster_center()

            # update visit frequency
            recent_clust.update_visit_freq()

            # update hour visit frequencies
            recent_clust.update_hr_visit_freq(hr=row[time_col].hour)

            # update cluster timestamp
            recent_clust.update_visit_times(time_col=time_col)
            continue

        # create new cluster and add  to cluster
        new_clust_id = recent_clust.clust_id + 1
        new_clust = Cluster(clust_id=new_clust_id, x=row[x], y=row[y], last_visit=row[time_col],
                            members=[current_member])
        # update visit frequency
        new_clust.update_visit_freq()

        # update hour visit frequencies
        new_clust.update_hr_visit_freq(hr=row[time_col].hour)

        clusters.append(new_clust)

    return clusters


def generate_unique_locs(df=None, x=None, y=None):
    """
    Returns unique locations visited (e.g., on a single day)
    :param oneday_df:
    :return:
    """
    df_unique_locs = df.drop_duplicates([x, y])
    # in case Lat, Lon are strings, convert to float
    df_unique_locs[y] = df[y].astype('float64')
    df_unique_locs[x] = df[y].astype('float64')

    unique_locs = []
    for idx, row in df_unique_locs.iterrows():
        unique_locs.append(POINT(point_id=idx, x=row[x], y=row[x], freq=-1))

    return unique_locs


def calculate_distance(pt1=None, pt2=None):
    """
    Computes distance between two geographic coordinates
    :param pt1: [Lat,Lon] for first point
    :param pt2: [Lat,Lon] for second
    :returns distance in km between the two points
    """
    # Radius of the earth in km (Hayford-Ellipsoid)
    EARTH_RADIUS = 6378388 / 1000

    d_lat = radians(pt1[0] - pt2[0])
    d_lon = radians(pt1[1] - pt2[1])

    lat1 = radians(pt1[0])
    lat2 = radians(pt2[0])

    a = sin(d_lat / 2) * sin(d_lat / 2) + \
        sin(d_lon / 2) * sin(d_lon / 2) * cos(lat1) * cos(lat2)

    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return c * EARTH_RADIUS


def add_weekdays(row, day_dict=None):
    return day_dict[row['wkday']]


def combine_selected_csv_files(folder_with_csv_files=None, number_to_save=None, out_csv_file=None):
    """
    Save a sample of the small CSV files into a CSV file for exploration
    :param folder_with_csv_files:
    :param number_to_save:
    :return:
    """

    # get a list of CSV file using os module listdir() function
    files = os.listdir(folder_with_csv_files)

    # create a list to hold pandas dataframes
    df_lst = []
    cnt = 0
    for f in files:
        if f.endswith('csv'):
            fpath = os.path.join(folder_with_csv_files, f)
            df = pd.read_csv(fpath)
            df_lst.append(df)
            cnt += 1
            if cnt % 100 == 0:
                print('Done with {} files so far'.format(cnt))

            if cnt == number_to_save:
                break

    df = pd.concat(df_lst)
    df.to_csv(out_csv_file, index=False)


def preprocess_cdrs_using_spark(file_or_folder=None, number_of_users_to_sample=None,
                                output_csv=None, date_format='%Y%m%d%H%M%S',
                                debug_mode=True, loc_file=None, save_to_csv=False):
    """
    In this function, we perfom some basic preprocessing such as below:
    1. rename columns
    2. change some data types
    3. Add location details
    Eventually, we will sample the data to use for our analysis
    :param data_folder:
    :param output_csv_for_sample_users:
    :return:
    """

    # create SparkSession object
    spark = SparkSession.builder.master("local[8]").appName("data_processor").getOrCreate()

    # read data with spark
    df = spark.read.csv(path=file_or_folder, header=True)

    # repartition to speed up
    df = df.repartition(10)

    # if just testing/debugging, pick only a small dataset
    if debug_mode:
        dfs = df.sample(fraction=0.001)
        df = dfs

    # rename columns to remove space and replace with underscore
    df2 = (df.withColumnRenamed("cdr datetime", "cdrDatetime")
        .withColumnRenamed("calling phonenumber2", "phoneNumber")
        .withColumnRenamed("last calling cellid", "cellId")
        .withColumnRenamed("call duration", "cellDuration"))

    # drop cdr type column
    df3 = df2.drop('cdr type')

    # Use Spark UDF to add date and datetime
    add_datetime = udf(lambda x: datetime.strptime(x, date_format), TimestampType())
    add_date = udf(lambda x: datetime.strptime(x, date_format), DateType())

    # create timestamp
    df4 = df3.withColumn('datetime', add_datetime(col('cdrDatetime')))
    df5 = df4.withColumn('date', add_date(col('cdrDatetime')))

    # lets make sure we dont have any null phoneNumbers
    df6 = df5.filter(df5['phoneNumber'].isNotNull())

    # Lets merge with location details using cellId from CDRs and also
    # cellID on the other
    dfLoc = pd.read_csv(loc_file)
    dfLoc.rename(columns={'cell_id': 'cellId'}, inplace=True)
    sdfLoc = spark.createDataFrame(dfLoc)
    df7 = df6.join(sdfLoc, on='cellId', how='inner')

    # select nsample users to work with
    all_users = df7.select('phoneNumber').distinct().collect()

    # randomly select users using filter statement
    random_user_numbers = [i['phoneNumber'] for i in random.choices(all_users, k=number_of_users_to_sample)]

    # select only our random user data
    dfu = df7.filter(df7['phoneNumber'].isin(random_user_numbers))

    # save to CSV if necessary
    if save_to_csv:
        dfu.coalesce(1).write.csv(path=output_csv, header=True)
    else:
        return dfu


def explore_data(df=None, output_plot_file=None, output_heatmap=None):
    """
    For quick examination of user activity, lets generate
    user call count and do a simple plot.
    """
    # Number of days in data
    dates_rows = df.select('date').distinct().collect()
    sorted_dates = sorted([i['date'] for i in dates_rows])
    diff = sorted_dates[-1] - sorted_dates[0]
    num_days = diff.days

    # call count by hour
    add_hr = udf(lambda x: x.hour, IntegerType())
    add_wkday = udf(lambda x: x.weekday(), IntegerType())
    day_dict = {0: 'Mon', '1': 'Tue', '2': 'Wed', 3: 'Thurs', 4: 'Frid', 5: 'Sat', 6: 'Sun'}

    dfHr = df.withColumn('hr', add_hr(col('datetime')))
    dfHr2 = dfHr.withColumn('wkday', add_wkday(col('datetime')))
    dfWkDay = dfHr2.groupBy('wkday', 'hr').count().toPandas()
    dfWkDay['weekDay'] = dfWkDay.apply(add_weekdays, args=(day_dict,), axis=1)
    dfWkDay.drop(labels=['wkday'], axis=1, inplace=True)
    dfWkDayPivot = dfWkDay.pivot(index='weekDay', columns='hr', values='count')
    d = dfWkDayPivot.reset_index()
    # ax = sns.heatmap(d)
    # ax.get_figure().savefig(output_heatmap)

    # group user and count number of events
    # convert resulting spark dataframe to pandas
    dfGroup = df.groupBy('phoneNumber').count().toPandas()

    # create a distribution plot of user call count using
    # seaborn
    ax = sns.distplot(dfGroup['count'])

    # save plot as png file
    ax.get_figure().savefig(output_plot_file)

    # report average number calls per day for each user
    dfGroupDay = df.groupBy('phoneNumber', 'date').count().toPandas()

    # get mean and median
    mean = dfGroupDay['count'].mean()
    median = dfGroupDay['count'].median()

    # data duration
    return mean, median, num_days, d

def generate_trips_by_day(df=None, datecol=None, lon=None, lat=None):
    """
    Generate trips for all given days. Also sets other user attributes
    :return:
    """

    total_uniq_xy = generate_unique_locs(df=df, x=lon, y=lat)
    dates = sorted(list(df.date.unique()))
    num_of_days_with_trips = 0  # record number of days with trips
    dates_trps = {d: None for d in dates}  # record trips finished on that day
    unique_locs_by_day = {d: None for d in dates}  # record unique locations visited each day
    dates_dist = {k: 0 for k in list(df['date'].unique())}  # distances travelled each day
    dates_clusters = {d: None for d in dates}

    for d, t in dates_trps.items():
        # data for this day
        dfd = df[df['date'] == d]

        # get number of unique locations for this day
        uniq_xy = generate_unique_locs(df=dfd, x=lon, y=lat)
        unique_locs_by_day[d] = len(uniq_xy)

        # generate clusters with stay time
        clusters = cluster_cells_within_radius_trps_version(loc_history=dfd, time_col=datecol, x=lon, y=lat)

        clusters_with_stay_time = generate_cluster_stay_time(clusters=clusters)

        # generate trips
        trps = detect_trips(clusters=clusters_with_stay_time)

        if not trps:
            continue

        # add to list of trips
        dates_trps[d] = trps

        # increment number of trip days
        num_of_days_with_trips += 1

        # add to list of clusters
        dates_clusters[d] = clusters_with_stay_time

        # distances traveled this day
        dist_mtx = distances_travelled(unique_locs=uniq_xy)
        dates_dist[d] = np.max(dist_mtx)

    # set attributes
    user_attributes = {'userId': df.iloc[0].phoneNumber,
                       'usageDays': len(dates),
                       'tripDays': num_of_days_with_trips,
                       'avgUniqLocsPerday': np.mean(list(unique_locs_by_day.values())),
                       'medianUniqLocsPerday': np.median(list(unique_locs_by_day.values())),
                       'totalUniqLocations': len(total_uniq_xy),
                       'medianFarthestDistance': np.median(list(dates_dist.values())),
                       'avgFarthestDistance': np.mean(list(dates_dist.values()))}

    return user_attributes


def generate_user_attributes_with_pandas(df_all_users=None, num_of_users=None, out_csv=None):
    """
    Loop through each user and generate their attributes
    :param df_all_users:
    :param num_of_users:
    :param out_csv:
    :return:
    """
    # create a copy of df to avoid annoying errors
    df = df_all_users.copy(deep=True)

    user_data = []
    user_list = list(df.phoneNumber.unique())

    for i, user in enumerate(user_list):
        udf = df[df.phoneNumber == user]
        data_pt = generate_trips_by_day(df=udf, datecol='datetime', lon='lon', lat='lat')
        user_data.append(data_pt)

        if i == num_of_users:
            break

    df = pd.DataFrame(user_data)
    df.to_csv(out_csv, index=False)

    return df


if __name__ == '__main__':
    dataFolder = '../../day5-case-studies/cdrs-test/'
    dfs = preprocess_cdrs_using_spark(file_or_folder=dataFolder, number_of_users_to_sample=1000,
                                      date_format='%Y%m%d%H%M%S',
                                      loc_file='../../day5-case-studies/cellTowers/staggered-cell-locs.csv',
                                      save_to_csv=False, debug_mode=False)
    pdf = dfs.toPandas()

    # generate_user_attributes_with_pandas(df_all_users=pdf, num_of_users=10, out_csv=users)
