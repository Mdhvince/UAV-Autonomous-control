import configparser
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

from planning.minimum_snap import MinimumSnap


class RRT:
    """
    Rapidly-exploring Random Tree (RRT) algorithm
    """
    def __init__(self, space_limits, start, goal, max_distance, max_iterations, obstacles=None):
        self.all_nodes = [start]
        self.goal = goal
        self.space_limits = space_limits
        self.max_distance = max_distance
        self.max_iterations = max_iterations
        self.obstacles = obstacles

        self.random_node = None
        self.nearest_node = None
        self.distance_of_nearest_node = None
        self.connected_nodes = []

    def run(self):
        """
        Run the RRT algorithm
        """
        for i in range(self.max_iterations):
            self._generate_random_node()
            self._find_nearest_node()

            self._update_tree()
            self.connected_nodes.append([self.nearest_node, self.random_node])

            if self._is_path_found() and self._is_valid_connection():
                break


    def _generate_random_node(self):
        """
        Generate a random node in the configuration space
        """
        x_rand = np.random.uniform(0, self.space_limits[0])
        y_rand = np.random.uniform(0, self.space_limits[1])
        z_rand = np.random.uniform(0, self.space_limits[2])

        self.random_node = np.array([x_rand, y_rand, z_rand])

    def _find_nearest_node(self):
        """
        Find the nearest node in the tree to the random node
        """
        # calculate the distance between the random node and all nodes in the tree
        distances = []
        for node in self.all_nodes:
            distances.append(np.linalg.norm(self.random_node - node))

        # find the nearest node
        self.nearest_node = self.all_nodes[np.argmin(distances)]
        self.distance_of_nearest_node = np.min(distances)


    def _is_valid_connection(self):
        """
        Check if the connection between the nearest node and the random node is collision-free
        """
        if self.obstacles is None:
            return True

        # check if the line connecting the nearest node and the random node intersects with any of the obstacles
        for obstacle in self.obstacles:
            xmin, xmax, ymin, ymax, zmin, zmax = obstacle
            node1, node2 = self.nearest_node, self.random_node


            direction = node2 - node1
            # Calculate the parameter t for the line equation: line = node1 + t * direction
            t = np.linspace(0, 1, 100)

            # Points along the line
            # points = node1 + np.outer(t, direction)
            points = np.outer(t, direction) + node1


            # Check if any of the points lie within the obstacle
            if np.any((points[:, 0] >= xmin) & (points[:, 0] <= xmax) &
                      (points[:, 1] >= ymin) & (points[:, 1] <= ymax) &
                      (points[:, 2] >= zmin) & (points[:, 2] <= zmax)):
                return False

        return True


    def _update_tree(self):
        """
        Update the tree with the new node
        """
        # if the random node is close enough to the nearest node, add it to the tree
        if (self.distance_of_nearest_node <= self.max_distance) and self._is_valid_connection():
            self.all_nodes.append(self.random_node)
        else:
            self.__generate_node_at_max_distance()

            if self._is_valid_connection():
                self.all_nodes.append(self.random_node)

    def __generate_node_at_max_distance(self):
        """
        Generate a node at max distance from the nearest node in the case that the random node is too far away
        """
        # generate a node at max distance from the nearest node on the line connecting the two nodes
        self.random_node = self.nearest_node + (self.random_node - self.nearest_node) * self.max_distance / self.distance_of_nearest_node


    def _is_path_found(self):
        """
        Find the node in the tree that is closest to the goal
        """
        # calculate the distance between the goal and all nodes in the tree
        distances = []
        for node in self.all_nodes:
            distances.append(np.linalg.norm(self.goal - node))

        # find the nearest node
        self.nearest_node_to_goal = self.all_nodes[np.argmin(distances)]

        # if the nearest node to the goal is within the max distance, we have found a path, return true
        if np.min(distances) <= self.max_distance:
            self.all_nodes.append(self.goal)
            self.connected_nodes.append([self.nearest_node_to_goal, self.goal])
            return True
        else:
            return False


    def get_path(self):
        """
        Get the path from the start to the goal
        """
        path = [self.goal]
        node = self.goal

        while not np.array_equal(node, self.all_nodes[0]):  # while the node is not the start node
            for connected_node in self.connected_nodes:
                if np.array_equal(connected_node[1], node):
                    path.append(connected_node[0])
                    node = connected_node[0]
                    break

        path.reverse()
        path = np.array(path)
        return path





if __name__ == "__main__":
    start = np.array([2, 2, 2])
    goal = np.array([10, 10, 10])
    max_distance = 3
    max_iterations = 5000

    obstacles = np.array([[0, 3, 3, 5, 0, 10], [3, 6, 3, 5, 0, 10]])  # xmin, xmax, ymin, ymax, zmin, zmax

    rrt = RRT([10, 10, 10], start, goal, max_distance, max_iterations, obstacles)
    rrt.run()

    fig = plt.figure(figsize=(32, 18))
    ax = fig.add_subplot(111, projection='3d')


    # plot the start and goal points in red and green respectively
    ax.scatter(start[0], start[1], start[2], c='r', marker='o')
    ax.scatter(goal[0], goal[1], goal[2], c='g', marker='o', s=100, alpha=0.5)

    # plot the nodes in the tree
    for node in rrt.all_nodes:
        ax.scatter(node[0], node[1], node[2], color='b', s=2, alpha=0.5)

    # plot the connected nodes
    for node in rrt.connected_nodes:
        nearest, random = node
        ax.plot(
            [nearest[0], random[0]],
            [nearest[1], random[1]],
            [nearest[2], random[2]], color='k', alpha=0.3)

    # plot the path
    path = rrt.get_path()
    for i in range(len(path) - 1):
        ax.plot(
            [path[i][0], path[i + 1][0]],
            [path[i][1], path[i + 1][1]],
            [path[i][2], path[i + 1][2]], color='r', alpha=0.5)


    # plot obstacles
    x, y, z = np.indices((11, 11, 11))  # space limits where cuboids can be placed

    for obstacle in obstacles:
        x_min, x_max = obstacle[0], obstacle[1]
        y_min, y_max = obstacle[2], obstacle[3]
        z_min, z_max = obstacle[4], obstacle[5]


        # represent the cuboid as a binary array
        cube = np.logical_and.reduce((
            x_min <= x, x < x_max,  # x is between x_min and x_max
            y_min <= y, y < y_max,  # y_min <= y <= y_max
            z_min <= z, z < z_max  # z_min <= z <= z_max
        ))

        colors = np.empty(cube.shape, dtype=object)
        colors[cube] = 'gray'
        ax.voxels(cube, facecolors=colors, alpha=.7)

    config = configparser.ConfigParser(inline_comment_prefixes="#")
    config_file = Path("/home/medhyvinceslas/Documents/programming/quad3d_sim/config.ini")
    config.read(config_file)

    T = MinimumSnap(config, "flight")
    T.coord_obstacles = None
    T.waypoints = path
    T.velocity = 2
    desired_trajectory = T.get_trajectory()[:, :3]

    ax.plot(
        desired_trajectory[:, 0],
        desired_trajectory[:, 1],
        desired_trajectory[:, 2], color='g', alpha=0.3, linewidth=3)


    plt.show()





