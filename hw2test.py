import requests
import os
import matplotlib.pyplot as plt
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, desc, hour, to_timestamp
import pandas as pd

# Initialize Spark session
spark = SparkSession.builder \
    .appName("NYC Taxi Data Analysis") \
    .config("spark.driver.extraJavaOptions", "-Djava.security.manager=allow") \
    .getOrCreate()

def download_file(url, local_path):
    """
    Helper function to download files from a URL.
    :param url: URL of the file
    :param local_path: Local path to save the file
    """
    response = requests.get(url)
    with open(local_path, "wb") as f:
        f.write(response.content)

    # Verify the file exists
    if not os.path.exists(local_path):
        raise FileNotFoundError(f"File {local_path} was not downloaded successfully.")


def join_look_up_with_cities(df, taxi_zone_lookup):
    """
    Join the taxi data with the zone lookup table and rename columns to avoid ambiguity.
    :param df: Dataframe
    :param taxi_zone_lookup: Dataframe
    :return: Dataframe
    """
    # Join for Pickup Zone
    df = df.join(taxi_zone_lookup, df.PULocationID == taxi_zone_lookup.LocationID, "left") \
        .withColumnRenamed("Zone", "Pickup_Zone") \
        .withColumnRenamed("Borough", "Pickup_Borough") \
        .withColumnRenamed("service_zone", "Pickup_Service_Zone") \
        .drop("LocationID")

    # Join for Dropoff Zone
    df = df.join(taxi_zone_lookup, df.DOLocationID == taxi_zone_lookup.LocationID, "left") \
        .withColumnRenamed("Zone", "Dropoff_Zone") \
        .withColumnRenamed("Borough", "Dropoff_Borough") \
        .withColumnRenamed("service_zone", "Dropoff_Service_Zone") \
        .drop("LocationID")

    return df


def find_statistical_information(df, dataset_name):
    """
    Retrieve statistical information (You can use aggregate functions e.g max, average of columns)
    :param df: Dataframe
    :param dataset_name: Name of the dataset (e.g., "Yellow Taxi", "Green Taxi")
    """
    print(f"Statistical Data for {dataset_name}:")
    df.describe().show()


def get_most_expensive_route(df, dataset_name):
    """
    What is the most expensive root for each datasets?
    :param df: Dataframe
    :param dataset_name: Name of the dataset (e.g., "Yellow Taxi", "Green Taxi")
    """
    most_expensive = df.orderBy(col("total_amount").desc()).first()
    print(f"Most Expensive Route in {dataset_name}:")
    print(
        f"Pickup Zone: {most_expensive['Pickup_Zone']}, Dropoff Zone: {most_expensive['Dropoff_Zone']}, Total Amount: {most_expensive['total_amount']}")


def get_busiest_taxi_station(df, dataset_name):
    """
    What is the busiest Taxi station for each datasets?
    :param df: Dataframe
    :param dataset_name: Name of the dataset (e.g., "Yellow Taxi", "Green Taxi")
    """
    busiest_station = df.groupBy("Pickup_Zone").agg(count("*").alias("Trip_Count")) \
        .orderBy(desc("Trip_Count")) \
        .first()
    print(
        f"Busiest Taxi Station in {dataset_name}: {busiest_station['Pickup_Zone']} with {busiest_station['Trip_Count']} trips")


def get_top_5_busiest_area(df, dataset_name):
    """
    What is the top 5 busiest Area for each datasets?
    :param df: Dataframe
    :param dataset_name: Name of the dataset (e.g., "Yellow Taxi", "Green Taxi")
    """
    top_5_busiest = df.groupBy("Pickup_Zone").agg(count("*").alias("Trip_Count")) \
        .orderBy(desc("Trip_Count")) \
        .limit(5)
    print(f"Top 5 Busiest Areas in {dataset_name}:")
    top_5_busiest.show()


def get_longest_trips(df, dataset_name):
    """
    What is the longest trip in each dataset?
    :param df: Dataframe
    :param dataset_name: Name of the dataset (e.g., "Yellow Taxi", "Green Taxi")
    """
    longest_trip = df.orderBy(col("trip_distance").desc()).first()
    print(f"Longest Trip in {dataset_name}:")
    print(
        f"Pickup Zone: {longest_trip['Pickup_Zone']}, Dropoff Zone: {longest_trip['Dropoff_Zone']}, Trip Distance: {longest_trip['trip_distance']}")


def get_crowded_places_per_hour(df, dataset_name):
    """
    Find the crowded Pickup and Drop-off zones for each hour.
    :param df: Dataframe
    :param dataset_name: Name of the dataset (e.g., "Yellow Taxi", "Green Taxi")
    """
    # Check if the dataset is Yellow Taxi or Green Taxi
    if "tpep_pickup_datetime" in df.columns:
        pickup_col = "tpep_pickup_datetime"
        dropoff_col = "tpep_dropoff_datetime"
    elif "lpep_pickup_datetime" in df.columns:
        pickup_col = "lpep_pickup_datetime"
        dropoff_col = "lpep_dropoff_datetime"
    else:
        raise ValueError("Dataset does not contain pickup/dropoff datetime columns.")

    # Extract hour from pickup and dropoff datetime
    df = df.withColumn("Pickup_Hour", hour(to_timestamp(col(pickup_col))))
    crowded_pickup_zones = df.groupBy("Pickup_Hour", "Pickup_Zone").agg(count("*").alias("Pickup_Count")) \
        .orderBy(desc("Pickup_Count"))

    df = df.withColumn("Dropoff_Hour", hour(to_timestamp(col(dropoff_col))))
    crowded_dropoff_zones = df.groupBy("Dropoff_Hour", "Dropoff_Zone").agg(count("*").alias("Dropoff_Count")) \
        .orderBy(desc("Dropoff_Count"))

    print(f"Crowded Pickup Zones per Hour in {dataset_name}:")
    crowded_pickup_zones.show()

    print(f"Crowded Dropoff Zones per Hour in {dataset_name}:")
    crowded_dropoff_zones.show()


def get_busiest_hours(df, dataset_name):
    """
    Find the Pickup and Drop-off count for each hour. After that draw two lineplot graphs for Pickup and Drop-off.
    This function returns busiest hour that you will use it in draw_busiest_hours_graph to draw lineplot graph.
    :param df: Dataframe
    :param dataset_name: Name of the dataset (e.g., "Yellow Taxi", "Green Taxi")
    :return: Dataframe
    """
    # Check if the dataset is Yellow Taxi or Green Taxi
    if "tpep_pickup_datetime" in df.columns:
        pickup_col = "tpep_pickup_datetime"
        dropoff_col = "tpep_dropoff_datetime"
    elif "lpep_pickup_datetime" in df.columns:
        pickup_col = "lpep_pickup_datetime"
        dropoff_col = "lpep_dropoff_datetime"
    else:
        raise ValueError("Dataset does not contain pickup/dropoff datetime columns.")

    # Extract hour from pickup and dropoff datetime
    df = df.withColumn("Pickup_Hour", hour(to_timestamp(col(pickup_col))))
    pickup_counts = df.groupBy("Pickup_Hour").agg(count("*").alias("Pickup_Count")).orderBy("Pickup_Hour")

    df = df.withColumn("Dropoff_Hour", hour(to_timestamp(col(dropoff_col))))
    dropoff_counts = df.groupBy("Dropoff_Hour").agg(count("*").alias("Dropoff_Count")).orderBy("Dropoff_Hour")

    return pickup_counts, dropoff_counts


def draw_busiest_hours_graph(pickup_counts, dropoff_counts, dataset_name):
    """
    You will use get_busiest_hours' result here. With this dataframe you should draw hour to count lineplot.
    :param pickup_counts: Dataframe
    :param dropoff_counts: Dataframe
    :param dataset_name: Name of the dataset (e.g., "Yellow Taxi", "Green Taxi")
    """
    pickup_counts_pd = pickup_counts.collect()  # Collect data
    dropoff_counts_pd = dropoff_counts.collect()  # Collect data

    # Convert to Pandas DataFrame
    pickup_counts_pd = pd.DataFrame(pickup_counts_pd, columns=['Pickup_Hour', 'Pickup_Count'])
    dropoff_counts_pd = pd.DataFrame(dropoff_counts_pd, columns=['Dropoff_Hour', 'Dropoff_Count'])

    plt.figure(figsize=(10, 5))
    plt.plot(pickup_counts_pd['Pickup_Hour'], pickup_counts_pd['Pickup_Count'], label='Pickup Count')
    plt.plot(dropoff_counts_pd['Dropoff_Hour'], dropoff_counts_pd['Dropoff_Count'], label='Dropoff Count')
    plt.xlabel('Hour of the Day')
    plt.ylabel('Count')
    plt.title(f'Pickup and Dropoff Counts per Hour - {dataset_name}')
    plt.legend()
    plt.show()


def get_tip_correlation(df, dataset_name):
    """
    (BONUS) What do other columns affect the tip column? (HINT: Correlation of tip amount to other columns)
    :param df: Dataframe
    :param dataset_name: Name of the dataset (e.g., "Yellow Taxi", "Green Taxi")
    """
    correlation = df.stat.corr("tip_amount", "total_amount")
    print(f"Correlation between tip_amount and total_amount for {dataset_name}: {correlation}")


if __name__ == '__main__':
    # URLs for the datasets
    yellow_taxi_url = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2021-03.parquet"
    green_taxi_url = "https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2021-03.parquet"
    taxi_zone_lookup_url = "https://d37ci6vzurychx.cloudfront.net/misc/taxi+_zone_lookup.csv"

    # Define the directory to save the files
    download_dir = os.path.join(os.getcwd(), "data")
    os.makedirs(download_dir, exist_ok=True)

    # Download the datasets
    yellow_taxi_path = os.path.join(download_dir, "yellow_tripdata_2021-03.parquet")
    green_taxi_path = os.path.join(download_dir, "green_tripdata_2021-03.parquet")
    taxi_zone_lookup_path = os.path.join(download_dir, "taxi_zone_lookup.csv")

    download_file(yellow_taxi_url, yellow_taxi_path)
    download_file(green_taxi_url, green_taxi_path)
    download_file(taxi_zone_lookup_url, taxi_zone_lookup_path)

    # Load the datasets into Spark DataFrames
    yellow_taxi_data = spark.read.parquet(yellow_taxi_path)
    green_taxi_data = spark.read.parquet(green_taxi_path)
    taxi_zone_lookup = spark.read.csv(taxi_zone_lookup_path, header=True, inferSchema=True)

    # Join datasets with zone lookup
    yellow_taxi_data = join_look_up_with_cities(yellow_taxi_data, taxi_zone_lookup)
    green_taxi_data = join_look_up_with_cities(green_taxi_data, taxi_zone_lookup)

    # Retrieve statistical information
    find_statistical_information(yellow_taxi_data, "Yellow Taxi")
    find_statistical_information(green_taxi_data, "Green Taxi")

    # Most expensive route
    get_most_expensive_route(yellow_taxi_data, "Yellow Taxi")
    get_most_expensive_route(green_taxi_data, "Green Taxi")

    # Busiest taxi station
    get_busiest_taxi_station(yellow_taxi_data, "Yellow Taxi")
    get_busiest_taxi_station(green_taxi_data, "Green Taxi")

    # Top 5 busiest areas
    get_top_5_busiest_area(yellow_taxi_data, "Yellow Taxi")
    get_top_5_busiest_area(green_taxi_data, "Green Taxi")

    # Longest trips
    get_longest_trips(yellow_taxi_data, "Yellow Taxi")
    get_longest_trips(green_taxi_data, "Green Taxi")

    # Crowded places per hour
    get_crowded_places_per_hour(yellow_taxi_data, "Yellow Taxi")
    get_crowded_places_per_hour(green_taxi_data, "Green Taxi")

    # Busiest hours
    yellow_pickup_counts, yellow_dropoff_counts = get_busiest_hours(yellow_taxi_data, "Yellow Taxi")
    green_pickup_counts, green_dropoff_counts = get_busiest_hours(green_taxi_data, "Green Taxi")

    # Draw busiest hours graph
    draw_busiest_hours_graph(yellow_pickup_counts, yellow_dropoff_counts, "Yellow Taxi")
    draw_busiest_hours_graph(green_pickup_counts, green_dropoff_counts, "Green Taxi")

    # Tip correlation
    get_tip_correlation(yellow_taxi_data, "Yellow Taxi")
    get_tip_correlation(green_taxi_data, "Green Taxi")

    # Clean up temporary files (optional)
    os.remove(yellow_taxi_path)
    os.remove(green_taxi_path)
    os.remove(taxi_zone_lookup_path)