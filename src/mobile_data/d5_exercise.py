"""
GENERAL INSTRUCTIONS
WARNING: For Python beginners:
the instructions here will only make sense after you have gone through and
completed the training materials.

1. WHICH PART TO CHANGE?: Uncomment every line with  [YOUR CODE HERE] and replace it with your code.
Please don't change anything else other than these lines.

2. USE OF JUPYTER NOTEBOOK: For those who would like to use Jupyter Notebook. You can copy and paste
each function in the notebook environment, test your code their. However,
remember to paste back your code in a .py file and ensure that its running
okay.

3. IDENTATION: Please make sure that you check your identation

4. Returning things frm function: All the functions below have to return a value.
Please dont forget to use the return statement to return a value.

5. HINTS: please read my comments for hints and instructions where applicable

6. DEFINING YOUR OWN FUNCTIONS: where I ask you to define your own function
please make sure that you name the function exactly as I said.

7. Please work on the following functions in the order provided:
    - combine_selected_csv_files
    - preprocess_cdrs_using_spark
    - add_weekdays
    - explore_data
    - generate_user_attributes_with_pandas
"""




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
    given a dataframe of user data.
    :param df: user data
    :return:
    """
    # use drop_duplicates() to drop duplicate x,y pairs
    # please check documenttion on how to use the function
    # df_unique_locs = YOUR CODE
    #
    # # in case Lat, Lon are strings, convert to float
    # df_unique_locs[y] = YOUR CODE
    # df_unique_locs[x] = YOUR CODE
    #
    # # declare an empty list to hold the unique locations
    # unique_locs = YOUR CODE
    # for idx, row in df_unique_locs.iterrows():
    #     unique_locs.append(POINT(point_id=idx, x=row[x], y=row[x], freq=-1))
    #
    # # return the unique locations
    # YOUR CODE



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
    """
    Given  a dictionary object like this: {1: 'Monday', 2:'Tuesday}
    This function should return the day of the week using the int
    key. We will use it in a pandas dataframe
    :param row:
    :param day_dict:
    :return:
    """

    # YOUR CODE HERE


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

    # # create a SparkSession object
    # spark = YOUR CODE
    #
    # # read data with spark
    # df = YOUR CODE
    #
    # # repartition to speed up
    # df = df.repartition(10)
    #
    # # if just testing/debugging, pick only a small dataset
    # # by using the sample function of spark
    # if debug_mode:
    #     dfs = YOUR CODE
    #     df = dfs
    #
    # # rename columns to remove space and name them using camel
    # # case  sytle like this: cdr datetime becomes cdrDatetime
    # # calling phonenumber becomes just phoneNumber
    # # if you are renaming more than one column, you can
    # # chain the commands and add round brackets like this:
    # df2 =    YOUR CODE
    #
    # # drop the 'cdr type' column
    # df3 = YOUR CODE
    #
    # # Use Spark UDF to add date and datetime
    # add_datetime = udf(lambda x: datetime.strptime(x, date_format), TimestampType())
    # add_date = udf(lambda x: datetime.strptime(x, date_format), DateType())
    #
    # # create timestamp
    # df4 = df3.withColumn('datetime', add_datetime(col('cdrDatetime')))
    # df5 = YOUR CODE
    #
    # # lets make sure we dont have any null phoneNumbers
    # # use spark filter() function to remove null phoneNumbers
    # df6 = YOUR CODE
    #
    # # Lets merge with location details using cellId from CDRs and also
    # # cellID on the other
    # # read pandas dataframe of location details
    # dfLoc = YOUR CODE
    #
    # # rename column 'cell_id' to 'cellId' in the pandas dataframe
    # YOUR CODE
    #
    # # create spark dataframe from the pandas dataframe
    # sdfLoc = YOUR CODE
    #
    # # join the cdrs dataframe with the location dataframe
    # df7 = YOUR CODE
    #
    # # create a list of unique user phoneNumbers
    # all_users = YOUR CODE
    #
    # # randomly select the required number of users
    # # using the random.choices() function
    # random_user_numbers = YOUR CODE
    #
    # # select only our random user data using spark filter
    # dfu = YOUR CODE
    #
    # # save to CSV if necessary
    # if save_to_csv:
    #     dfu.coalesce(1).write.csv(path=output_csv, header=True)
    # else:
    #     return dfu


def explore_data_with_spark(df=None, output_plot_file=None, output_heatmap=None):
    """
    Lets do a quick exploration of the data by generating the following:
    1. Number of days in the data
    2. User call count stats
    3. Weekday and hour calling patterns
    """
    # =====================================
    # Calculate the number of days in data
    # =====================================

    # # use relevant spark function to generate
    # # a list of unique dates, recall that the date
    # # column is 'date
    # dates_rows = YOUR CODE
    # # sort the dates using sorted() function
    # sorted_dates = YOUR CODE
    # # use list indexing to get the first element and last
    # # element from the sorted list, substract them to get
    # # time difference
    # diff = YOUR CODE
    # # use days function to get the number of days
    # num_days = YOUR CODE
    #
    # # =====================================
    # # Generate weekday and hour call count
    # # =====================================
    #
    # # define UDF to calculate hour and weekday
    # # for weekday use weekday() function while
    # # for hour, use hour
    # add_hr = YOUR CODE
    # add_wkday = YOUR CODE
    # # create a dictionary with keys as the weekday integers while the values
    # # are the weekday name
    # day_dict = YOUR CODE
    #
    # # add hour column, lets call it 'hr
    # # also add weekday column, we call it 'wkday'
    # dfHr = YOUR CODE
    # dfHr2 = YOUR CODE
    #
    # # use spark group by to counter number of calls by each
    # # hour of the day, the groupBy column will be 'wkday, hr'
    # dfWkDay = YOUR CODE
    #
    # # use the  add_weekdays  function to assign
    # dfWkDay['weekDay'] = dfWkDay.apply(add_weekdays, args=(day_dict,), axis=1)
    # dfWkDay.drop(labels=['wkday'], axis=1, inplace=True)
    # dfWkDayPivot = dfWkDay.pivot(index='weekDay', columns='hr', values='count')
    # d = dfWkDayPivot.reset_index()
    # ax = sns.heatmap(d)
    # ax.get_figure().savefig(output_heatmap)
    #
    # # =====================================
    # # Number of calls for each user
    # # =====================================
    # # group user and count number of events
    # # convert resulting spark dataframe to pandas
    # dfGroup = YOUR CODE
    #
    # # create a distribution plot of user call count using
    # # seaborn
    # ax = YOUR CODE
    #
    # # save plot as png file
    # YOUR CODE
    #
    # # report average number calls per day for each user
    # # first use spark groupBy on phoneNumber and day, then
    # # convert that object to pandas dataframe using toPandas()
    # # function
    # dfGroupDay = YOUR CODE
    #
    # # get mean and median
    # mean = YOUR CODE
    # median = YOUR CODE
    #
    # # return results like this mean, median, number of days
    # YOUR CODE
    #


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
    # # create a copy of df to avoid annoying errors
    # df = df_all_users.copy(deep=True)
    #
    # # declare an empty list
    # user_data = YOUR CODE
    #
    # # create a list of unique user IDs using the phoneNumber column
    # user_list = YOUR CODE
    #
    # for i, user in enumerate(user_list):
    #     # subset the data so that we have data for this user
    #     # only using phoneNumber
    #     udf = YOUR CODE
    #
    #     # use generate_trips_by_day function to
    #     # generate attributes for this user
    #     data_pt = YOUR CODE
    #
    #     # append the data_pt above to our user_data list
    #     YOUR CODE
    #
    #     # stop loop execution if we hit the required number of users
    #     if i == num_of_users:
    #         break
    #
    # # create dataframe and save it to CSV
    # YOUR CODE


def combine_selected_csv_files(folder_with_csv_files=None, number_to_save=None, out_csv_file=None):
    """
    Save a sample of the small CSV files into a CSV file for exploration.
    Please test this with very few files to avoid wasting time
    :param folder_with_csv_files:
    :param number_to_save:
    :return:
    """

    # # get a list of CSV file using os module listdir() function
    # files = YOUR CODE
    #
    # # create a list to hold pandas dataframes
    # df_lst = YOUR CODE
    #
    # #create a counter variable whcih will help you stop
    # # the loop when you reach the required number of files
    # cnt = YOUR CODE
    # for f in files:
    #     if f.endswith('csv'):
    #         fpath = os.path.join(folder_with_csv_files, f)
    #         df = YOUR CODE
    #         # append this df to the list of dfs above
    #         YOUR CODE
    #
    #         # increment the counter variable
    #         YOUR CODE
    #
    #         # stop the loop using break statement when you have
    #         # processes the required number of files
    #         # as defined by number_to_save
    #         YOUR CODE
    #
    # # use pandas function concat() like this: pd.concat()
    # # to concatenate all the dfs in the list
    # df = YOUR CODE
    #
    # # save your new dataframe
    # YOUR CODE
