#!/usr/bin/env python
"""
Script for finding group members and starting points of a group tour.
"""
import math
from typing import Optional

import click
import folium
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

# Constants:
# convert lat, lon coords to radian
deg2rad = math.pi / 180
# Earth radius in km
R = 6373
# Approximate tour length in km
tour_length = 50


def _osm_geojson_to_csv(
    path_geojson: str = "cafes.geojson", path_csv: str = "nodes.csv"
):

    """
    Just for reference, this function shows how OpenStreetMap geojson data was processed
    into the csv file nodes.csv.

    The cafe.geojson file was obtained from
    http://overpass-turbo.eu/ using the query
    ```
    node
    [amenity=cafe]
    (52.1, 12.5, 52.9, 14.5);
    out;
    ```
    """

    df = pd.read_json(path_geojson)

    # Turn dict into columns
    df = pd.json_normalize(df["features"])

    # Define and extract the columns we use
    df["longitude"] = df["geometry.coordinates"].apply(lambda x: x[0])
    df["latitude"] = df["geometry.coordinates"].apply(lambda x: x[1])
    df = df.rename(columns={"properties.name": "name"})

    # Remove nan's
    df = df.dropna(subset=["name", "longitude", "latitude"])
    df = df.reset_index(drop=True)

    # Export to csv
    print(f"Writing to {path_csv}")
    df[["id", "longitude", "latitude", "name"]].to_csv(path_csv, index=False)

    return


def _get_center_coord(df: pd.DataFrame):
    """
    Just for reference, this function shows how one could get a median-like
    average value from a set of starting points, e.g. from the same user_id,
    using a nearest-neighbor approach.
    It returns the coordinate which has the shortest mean distance to other
    coordinates. It uses the Haversine metric since this is the correct metric
    for points on a sphere in spherical coordinates.
    """
    if len(df) <= 2:
        # No point in finding the most central point with less than 3 points
        return df[["latitude", "longitude"]].iloc[0]
    X = df[["latitude", "longitude"]]
    nn = NearestNeighbors(n_neighbors=len(X), metric="haversine").fit(X)
    dist, _ = nn.kneighbors(X)
    imin = np.argmin(np.mean(dist, axis=1))
    return X.iloc[imin]


def _integer_to_hex_id(n: int, width: int = 5) -> str:
    """Convert an integer to a hexadecimal id of fixed width"""
    assert (
        len(hex(n).split("0x")[-1]) <= width
    ), "Given number {n} should fit within given width {width}"
    return f"{n:0>{width}x}"


def _run(
    user_csv: str, nodes_csv: str, out_csv: str = "results.csv", num_users: int = 30
) -> None:
    """Find starting point and group members for each user.

    The results are written to a csv file.

    Args:
    (str) user_csv: path to csv file containing user data. One row per user and starting point.
    (str) nodes_csv: path to csv file containing node data of potential meeting point.
        One row per node.
    (str) out_csv: path to csv file to write results to. Defaults to 'results.csv'.
    (int) num_users: number of group members to search for. Defaults to 30.

    """
    df_user = pd.read_csv(user_csv)
    df_nodes = pd.read_csv(nodes_csv)

    assert not df_user.isna().any().any(), "assuming user data has no null values"
    assert not df_nodes.isna().any().any(), "assuming node data has no null values"

    print("Number of unique users:", df_user["user_id"].nunique())
    print("Number of unique nodes:", df_nodes["id"].nunique())

    # Simplify by getting one location per user.
    # This is the simplest way of getting an average coordinate value.
    # One could imagine more rigorous approaches, see _get_center_coord().
    df_user = df_user.groupby("user_id").mean().reset_index()

    # Use given nodes, as candidate starting points. Here, nodes are locations of cafes.
    # Map each user location to the nearest node, the proposed starting point.
    # Convert angular coordinates to radians as expected by the Haversine metric.
    X_node = df_nodes[["latitude", "longitude"]].to_numpy() * deg2rad
    X_user = df_user[["latitude", "longitude"]].to_numpy() * deg2rad

    # Find the nodes using nearest neighbors
    nn = NearestNeighbors(n_neighbors=1, algorithm="auto", metric="haversine").fit(
        X_node
    )

    # Get the nearest node at the user locations
    # (indices here refer to the node data index)
    distances, indices = nn.kneighbors(X_user)

    # Convert distances from unitless to km
    distances = distances * R

    # Get relevant data of nearest-neighbor node
    nn_nodes = df_nodes.iloc[indices.ravel()][
        ["latitude", "longitude", "name"]
    ].reset_index()
    nn_nodes.columns = [c + "_node" for c in nn_nodes]

    # Nearest node with distance to corresponding user
    nn_nodes = pd.concat(
        [nn_nodes, pd.DataFrame(distances, columns=["distance_node"])], axis=1
    )

    # Add to user dataframe
    df_user = pd.concat([df_user, nn_nodes], axis=1)
    print(f"Selected {df_user['index_node'].nunique()} unique nodes as start points.")

    # Given now a starting point per user, find other users around that node.
    nn = NearestNeighbors(
        n_neighbors=num_users, algorithm="auto", metric="haversine"
    ).fit(X_user)

    # Get the nearest users at the selected node locations
    X_start_node = df_user[["latitude_node", "longitude_node"]].to_numpy() * deg2rad

    # Distances to starting point and indices of neighbors
    distances, indices = nn.kneighbors(X_start_node)

    assert not (
        np.diff(distances, axis=1) < 0
    ).any(), "Assumed distances are in ascending order"

    # Distances in units of km
    distances = distances * R

    # Neighbor user ids for each user_id
    user_ids = df_user["user_id"].to_numpy()[indices]

    # The neighbor list usually includes the current user.
    df_user["neighbors"] = pd.Series(user_ids.tolist())
    df_user["neighbor_distances"] = pd.Series(distances.tolist())

    # Filter out the distances and user_ids of  the current user from the group list
    # (other filtering could be performed here as well, e.g. based on the distance).
    df_user["potential_group_members"] = df_user.apply(
        lambda x: [uid for uid in x["neighbors"] if uid != x["user_id"]], axis=1
    )
    df_user["potential_group_member_distances"] = df_user.apply(
        lambda x: [
            dist
            for uid, dist in zip(x["neighbors"], x["neighbor_distances"])
            if uid != x["user_id"]
        ],
        axis=1,
    )

    # Add some stats on the user distances
    df_user["start_point_max_distance"] = df_user["neighbor_distances"].apply(np.max)

    # Print some metrics
    reasonable_distance = int(np.round(tour_length * 0.1))
    print(
        f"{(df_user['start_point_max_distance'] < reasonable_distance).mean()*100:.1f}% "
        f"of users have all potential group member at distances within {reasonable_distance} km "
        "of the proposed starting point."
    )

    # Prepare for output, final processing
    df_user = df_user.rename(
        columns=dict(
            distance_node="user_start_point_distance",
            latitude_node="start_point_latitude",
            longitude_node="start_point_longitude",
            name_node="start_point_name",
            latitude="user_avg_latitude",
            longitude="user_avg_longitude",
        )
    )

    # Add an id for the start point since it was requested.
    df_user["start_point_id"] = df_user["index_node"].apply(_integer_to_hex_id)
    # Convert the list into a string before writing to csv get the requested format.
    df_user["potential_group_members"] = df_user["potential_group_members"].apply(
        (lambda x: ",".join(x))
    )

    df_user = df_user[
        [
            "user_id",
            "user_avg_latitude",
            "user_avg_longitude",
            "start_point_id",
            "start_point_latitude",
            "start_point_longitude",
            "start_point_name",
            "user_start_point_distance",
            "potential_group_members",
            "potential_group_member_distances",
            "start_point_max_distance",
        ]
    ]
    print(f"Writing to {out_csv}")
    df_user.to_csv(out_csv, index=False)

    return


def visualize_user(
    df: pd.DataFrame, user_id: str, out_html: Optional[str] = None
) -> None:
    assert df.groupby("user_id").size().unique() == 1, "Expect one entry per user"
    row = df.query("user_id==@user_id").iloc[0]

    m = folium.Map(
        location=(row["user_avg_latitude"], row["user_avg_longitude"]),
        zoom_start=13,
        control_scale=True,
        width="80%",
        height="80%",
    )

    # Add marker for this user
    coord = row[["user_avg_latitude", "user_avg_longitude"]].values.ravel()
    folium.Marker(
        location=coord,
        tooltip="this user",
        popup=f"user_id: {user_id}, lat: {coord[0]}, lon: {coord[1]}",
        icon=folium.Icon(color="pink"),
    ).add_to(m)

    # Add marker for other users
    for uid_other in row["potential_group_members"]:
        # Look up coordinates
        coord = df.query("user_id == @uid_other")[
            ["user_avg_latitude", "user_avg_longitude"]
        ].values.ravel()
        folium.Marker(
            location=coord,
            tooltip="other user",
            popup=f"user_id: {uid_other}, lat: {coord[0]}, lon: {coord[1]}",
            icon=folium.Icon(color="blue"),
        ).add_to(m)

    # Add nearby cafe, start point node
    folium.Marker(
        location=[row["start_point_latitude"], row["start_point_longitude"]],
        tooltip="cafe start point",
        popup=(
            f"name: {row['start_point_name']}, "
            "lat: {row['start_point_latitude']}, lon: {row['start_point_longitude']}"
        ),
        icon=folium.Icon(color="green"),
    ).add_to(m)

    out_html = out_html if out_html is not None else f"user_id={user_id}.html"
    print(f"Writing to {out_html}")
    m.save(out_html)
    return


@click.group()
def cli():
    """
    This is a commandline tool for proposing personalized starting points
    and group members of a cycling tour.

    See the help section of each subcommand below for usage info.
    """
    pass


@click.command()
@click.option(
    "--user-data",
    default="tours.csv",
    help="Path to tours.csv file containing starting points of users",
    required=True,
    show_default=True,
)
@click.option(
    "--osm-data",
    default="nodes.csv",
    help="Path to nodes.csv file containing OpenStreetMap data data of potential meeting points.",
    required=True,
    show_default=True,
)
@click.option(
    "--results",
    default="out.csv",
    help="Path with results.",
    required=False,
    show_default=True,
)
@click.option(
    "--num-users",
    default=30,
    help="Number of potential group members to search for.",
    required=False,
    show_default=True,
)
def run(user_data, osm_data, results, num_users):
    """
    Search for user groups based on two input files: one containing user
    location data and the other containing OpenStreetMap node data with
    locations of starting point candidates.
    """
    print(f"Running {user_data}, {osm_data} {results}")

    _run(user_csv=user_data, nodes_csv=osm_data, out_csv=results, num_users=num_users)


@click.command()
@click.option(
    "--user-id",
    default=None,
    help="A user id from the user data set. If not given, will plot a random user.",
    required=False,
)
@click.option(
    "--results",
    default="out.csv",
    help="Path to result data",
    required=True,
    show_default=True,
)
@click.option("--output", default=None, help="Path for the output", required=False)
def viz(
    user_id,
    results,
    output,
):
    """
    Generate a visualization of the results for the given user id and write to file.

    Requires having run the algorithm.
    """
    df = pd.read_csv(results)
    if user_id is None:
        user_id = df["user_id"].sample().iloc[0]
        print("No user id provided. Plotting randomly selected user: ", user_id)
    # Turn string into list
    df["potential_group_members"] = df["potential_group_members"].apply(
        lambda x: x.split(",")
    )
    visualize_user(df=df, user_id=user_id, out_html=output)


if __name__ == "__main__":
    cli.add_command(run)
    cli.add_command(viz)
    cli()
