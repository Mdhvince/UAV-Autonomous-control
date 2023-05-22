import numpy as np

# rrt*
# generate random node
# find neighbors of random node within a radius
# among these neighbors, find the one with the lowest cost (min distance from the start node)
# connect the random node to the neighbor with the lowest cost

class RRT:
    """
    Rapidly-exploring Random Tree (RRT) algorithm
    """
    def __init__(self, space_limits, start, goal, max_distance, max_iterations, obstacles=None, use_star=False):
        self.all_nodes = [start]
        self.goal = goal
        self.space_limits = space_limits
        self.max_distance = max_distance
        self.max_iterations = max_iterations
        self.obstacles = obstacles
        self.use_star = use_star

        self.random_node = None
        self.candidate_node = None
        self.distance_of_candidate_node = None
        self.connected_nodes = []

    def run(self):
        """
        Run the RRT algorithm
        """
        for i in range(self.max_iterations):
            self._generate_random_node()
            self._find_candidate_node()

            self._update_tree()
            self.connected_nodes.append([self.candidate_node, self.random_node])

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


    def _find_candidate_node(self):
        """
        - Find the nearest node in the tree to the random node [RRT]
        - Find the neighbor with the lowest cost (min distance from the start node) [RRT*]
        """
        distances = []
        start_node = self.all_nodes[0]

        if self.use_star:
            neighbors = []
            costs = []

            # calculate the distance between the random node and all nodes in the tree
            for node in self.all_nodes:
                distances.append(np.linalg.norm(self.random_node - node))
            nearest_node = self.all_nodes[np.argmin(distances)]

            # place the random node at max distance from the nearest node if it is too far away but no connection
            distance_nearest = np.linalg.norm(self.random_node - nearest_node)
            if distance_nearest > self.max_distance:
                self.random_node = nearest_node + (self.random_node - nearest_node) * self.max_distance / distance_nearest

            # check if self.random_node contains infinite values
            assert np.all(np.isfinite(self.random_node)), "Random node contains infinite values"

            # find neighbors of random node within a radius
            for node in self.all_nodes:
                if np.linalg.norm(self.random_node - node) <= self.max_distance*2:
                    neighbors.append(node)

            # among these neighbors, find the one with the lowest cost (min distance from the start node)
            if len(neighbors) > 0:
                for node in neighbors:
                    costs.append(np.linalg.norm(start_node - node))

                self.candidate_node = neighbors[np.argmin(costs)]

                # here 0 just to say that it is close enough to the random node to validate the connection
                # because it is a neighbor
                self.distance_of_candidate_node = 0  # distance of the candidate node to the random node

            # else:
                # # nearest
                # for node in self.all_nodes:
                #     distances.append(np.linalg.norm(self.random_node - node))
                #
                # self.candidate_node = self.all_nodes[np.argmin(distances)]
                # self.distance_of_candidate_node = np.min(distances)

        else:
            # calculate the distance between the random node and all nodes in the tree
            for node in self.all_nodes:
                distances.append(np.linalg.norm(self.random_node - node))

            # find the nearest node
            self.candidate_node = self.all_nodes[np.argmin(distances)]
            self.distance_of_candidate_node = np.min(distances)  # distance of the candidate node to the random node


    def _is_valid_connection(self):
        """
        Check if the connection between the nearest node and the random node is collision-free
        """
        if self.obstacles is None:
            return True

        # check if the line connecting the nearest node and the random node intersects with any of the obstacles
        for obstacle in self.obstacles:
            xmin, xmax, ymin, ymax, zmin, zmax = obstacle
            node1, node2 = self.candidate_node, self.random_node


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
        # if the random node is close enough to the candidate node, add it to the tree
        if (self.distance_of_candidate_node <= self.max_distance) and self._is_valid_connection():
            self.all_nodes.append(self.random_node)
        else:
            if not self.use_star:
                self.__generate_node_at_max_distance()
                if self._is_valid_connection():
                    self.all_nodes.append(self.random_node)


    def __generate_node_at_max_distance(self):
        """
        Generate a node at max distance from the nearest node in the case that the random node is too far away
        """
        # generate a node at max distance from the nearest node on the line connecting the two nodes
        self.random_node = self.candidate_node + (self.random_node - self.candidate_node) * self.max_distance / self.distance_of_candidate_node


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

        # if the nearest node to the goal is within 1m, we have found a path, return true
        if np.min(distances) <= 1:
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
    pass





