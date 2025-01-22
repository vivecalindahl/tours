# About
This directory contains my solution for a mini-project carried out in a data scientist interview process with a company making a route planner app.
The task was to for each user suggest other users to connect with and a starting point for a 50 km group cycling ride.


# Get started
To try it out, set up a python environment supporting the Python requirements. The dependencies to install are specified in `requirements.txt`. Alternatively, if you use `pipenv` for managing virtual environmens, a `Pipfile` and corresponding `Pipfile.lock` file is ready at your disposal.

The solution is implemented in the executable `tours.py`. To get more info about it run
```
./tours.py --help
```

There are two subcommands, `run`, which runs the algorithm, and `viz`, which generates a visualization of the results for a single user. For more info run
```
./tours.py run --help

# or

./tours.py viz --help
```


# The approach
Besides the provided user data I have added an addtional csv file called `nodes.csv`, containing OpenStreetMap data of cafe locations. This list of "nodes" act as start point candidates. The rationale behind not using the locations provided in the user data  is primarily that I imagined that the start point of a solo ride is not necessarily ideal as a start point for a social ride.
For instance, it might start right outside the user's home, which could be an awkward proposal, or in the middle of a park with no suitable landmark for finding each other with. Therefor I opted for defining the start points as cafe's, which also makes it possible and natural to start out or end the ride with a snack or drink and getting to know each other.

Given these potential start points, I define the proposed start point per user as the one closest to the user.
To make the definition of "closest" easier, I map each user to a single location by averaging over the start points per user. In this dataset, the typical user has only 1 start point (the median value) so this averaging has no significant effect.

Finally, with the nearest cafe node as the origin of the to-be group ride, I form the potential group members as the `n` closest users to the start point. The group size `n` can be varied in the command line tool, but by default is set to the relatively large number 30 since it's expected that often users won't accept the invitation.


# The output

The results of the `run` command are written to a csv file with the following columns.

* `user_id`: user id
* `user_avg_latitude`: average latitude of this user, averaged over available starting points of the user.
* `user_avg_longitude`: average longitude of this user
* `start_point_id`: id of the start point
* `start_point_latitude`: latitude of the start point
* `start_point_longitude`: longitude of the start point
* `start_point_name`: name of the start point
* `user_start_point_distance`: the distance between the user and the start point in units of km.
* `potential_group_members`: comma-separated string of potential group members,
    not inlcuding the current user, in increasing order of distance to the start point
* `potential_group_member_distances`: comma-separated string of distances in units of km to the
    start point of the potential group members, in increasing order.
* `start_point_max_distance`: maximum distance of the group members average start point
    location to the proposed start point. This is worst-case metrics of distance if including all group members.


The `viz` command will save an html image to file. To view, open in a browser.


# Strengths, limitations, and possible extensions

The main strength of this approach is that it is simple, fast and easy to understand. For this dataset it also works fairly well, in the sense that most users of the proposed groups are within a reasonable distance of the start point. For instance, with user groups of  size 30 users,  99.6% of the group members have less than 5 km to add to the starting point relative to where they usually start.
Compared to a 50 km ride, this is a reasonable overhead.

A possible improvement here is to filter out users that are too far away from the proposed starting point. This could also be a post-processing step using the `potential_group_members` and `potential_group_member_distances` fields of the result dataset.
In addtion, In some cases the current user is too far from the start point even though it was selected as the closest one. This could be solved by adding more start point candidates, perhaps other types of nodes than cafe nodes considered here.

A weakness of using cafes as proposed start points is that the user may interpret it as advertisement being proposed a commercial establishment as a start point. Despite this I chose this because it would be my first choice of start point if I were to organize a group ride, for the reasons mentioned above. A better solution might use another set of non-commerical candidate nodes, such as historical landmarks or other typical meeting points.

Finally, given the same overall approach one could consider alternative ways to find the start point and group members of a specific user. For instance, instead of first finding a starting point given the user and then finding the group, I also considered the approach of first finding the group close to the user and then finding a start point close to the group (center). The advantage of that could be to avoid situations where a user at the boundary of the user distribution would "pick" a start point close to itself which is however quite far from other users located closer to the center. The alternative approach would instead pick a more balanced start point in the sense that it's closer to everyone in the group. In the end however, I favored the simplicity of the current approach.


# What I learned
What I found most fun about this work was to get some hands-on experience of working with and visualzing spatial user data. In particular I found and tested a couple of new libraries and applications. The `folium` visualization library made all the difference in getting a better feeling for the data when projecting it on a real city map. I was not familiar with OpenStreetMap api, but it was delighted to get a first taster in extracting the cafe node geojson dataset using OpenStreetMap tool http://overpass-turbo.eu/.