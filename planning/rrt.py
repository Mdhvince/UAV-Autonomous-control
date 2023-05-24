import numpy as np
from mayavi import mlab

class RRTStar:
    """
    Rapidly-exploring Random Tree (RRT) algorithm
    """
    def __init__(self, space_limits, start, goal, max_distance, max_iterations, obstacles=None):
        self.space_limits = space_limits
        self.start = start
        self.goal = goal
        self.step_size = max_distance
        self.max_iterations = max_iterations
        self.obstacles = obstacles
        self.epsilon = 0.1

        self.neighborhood_radius = 1.5 * max_distance
        self.all_nodes = [start]

        self.tree = {}

        assert self.neighborhood_radius > self.step_size, "Neighborhood radius must be larger than step size"

    def run(self):
        for it in range(self.max_iterations):
            new_node = self._generate_random_node()
            nearest_node = self._find_nearest_node(new_node)
            new_node = self._adapt_random_node_position(new_node, nearest_node)
            neighbors = self._find_valid_neighbors(new_node)
            best_neighbor = self._find_best_neighbor(neighbors)

            if best_neighbor is None:
                continue

            self._update_tree(best_neighbor, new_node)

            # among the neighbors, find if linking to the new node is better than the current parent (re-wire)
            self._rewire_safely(neighbors, new_node)

            # TODO: if path found, calculate the cost of the path and store it in a list. then continue. if the cost of
            #  the path is lower than the cost of the best path, update the best path and the best cost. if the cost of
            #  the path stay lower than the best cost for a certain number of iterations, stop the algorithm and return
            #  the best path

        if not self._is_path_found():
            raise Exception("No path found")


    def _generate_random_node(self):
        # with probability epsilon, sample the goal
        if np.random.uniform(0, 1) < self.epsilon:
            return self.goal

        x_rand = np.random.uniform(0, self.space_limits[0])
        y_rand = np.random.uniform(0, self.space_limits[1])
        z_rand = np.random.uniform(0, self.space_limits[2])
        random_node = np.round(np.array([x_rand, y_rand, z_rand]), 2)
        return random_node

    def _find_nearest_node(self, new_node):
        distances = []
        for node in self.all_nodes:
            distances.append(np.linalg.norm(new_node - node))
        nearest_node = self.all_nodes[np.argmin(distances)]
        return nearest_node

    def _adapt_random_node_position(self, new_node, nearest_node):
        """
        Adapt the random node position if it is too far from the nearest node
        """
        distance_nearest = np.linalg.norm(new_node - nearest_node)
        if distance_nearest > self.step_size:
            new_node = nearest_node + (new_node - nearest_node) * self.step_size / distance_nearest
            new_node = np.round(new_node, 2)
        return new_node

    def _find_valid_neighbors(self, new_node):
        neighbors = []
        for node in self.all_nodes:
            node_in_radius = np.linalg.norm(node - new_node) <= self.neighborhood_radius
            if node_in_radius and self._is_valid_connection(node, new_node):
                neighbors.append(node)
        return neighbors

    def _find_best_neighbor(self, neighbors):
        """
        Find the neighbor with the lowest cost. The cost is the distance from the start node to the neighbor
        """
        if len(neighbors) == 0:
            return None

        costs = []
        for neighbor in neighbors:
            cost = np.linalg.norm(neighbor - self.start)
            costs.append(cost)

        best_neighbor = neighbors[np.argmin(costs)]
        return best_neighbor

    def _update_tree(self, node, new_node):
        """
        Update the tree with the new node
        """
        # add the new node to the list of all nodes
        self.all_nodes.append(new_node)

        # add the new node to the tree
        node_key = str(np.round(new_node, 2).tolist())
        node_parent = np.round(node, 2)

        if not np.array_equal(node_parent, new_node):
            self.tree[node_key] = node_parent

    def _rewire_safely(self, neighbors, new_node):
        """
        Among the neighbors (without the already linked neighbor), find if linking to the new node is better than the
        current parent (re-wire).
        """
        for neighbor in neighbors:
            if np.array_equal(neighbor, self.tree[str(np.round(new_node, 2).tolist())]):
                # if the neighbor is already the parent of the new node, skip
                continue

            if self._is_valid_connection(neighbor, new_node):
                current_parent = self.tree[str(np.round(neighbor, 2).tolist())]
                current_cost = np.linalg.norm(current_parent - self.start)
                potential_new_cost = np.linalg.norm(new_node - self.start)

                if potential_new_cost < current_cost:
                    # if the new node is closer to the start node than the current parent, re-wire (the parent of the
                    # neighbor becomes the new node)
                    self.tree[str(np.round(neighbor, 2).tolist())] = new_node

    def _is_valid_connection(self, node, new_node):
        """
        Check if the connection between the candidate node and the new node (random or goal) is collision-free
        """
        if self.obstacles is None:
            return True

        # check if the line connecting the nearest node and the random node intersects with any of the obstacles
        for obstacle in self.obstacles:
            xmin, xmax, ymin, ymax, zmin, zmax = obstacle
            node1, node2 = node, new_node


            direction = node2 - node1
            # Calculate the parameter t for the line equation: line = node1 + t * direction
            t = np.linspace(0, 1, 100)

            # Points along the line
            points = np.outer(t, direction) + node1


            # Check if any of the points lie within the obstacle
            if np.any((points[:, 0] >= xmin) & (points[:, 0] <= xmax) &
                      (points[:, 1] >= ymin) & (points[:, 1] <= ymax) &
                      (points[:, 2] >= zmin) & (points[:, 2] <= zmax)):
                return False

        return True

    def _is_path_found(self):
        """
        Check if the goal node is in the tree as a child of another node
        """
        goal_node_key = str(np.round(self.goal, 2).tolist())
        return goal_node_key in self.tree.keys()


    def get_path(self):
        """
        Get the path from the goal node to the start node
        """
        path = [self.goal]
        node = self.goal
        while not np.array_equal(node, self.start):
            node = self.tree[str(np.round(node, 2).tolist())]
            path.append(node)

        return np.array(path[::-1]).reshape(-1, 3)







if __name__ == "__main__":

    start = np.array([0, 0, 0])
    goal = np.array([7, 7, 7])

    rrt = RRTStar(
        space_limits=np.array([10, 10, 10]),
        start=start,
        goal=goal,
        max_distance=1,
        max_iterations=500,
        obstacles=None,
    )
    rrt.run()

    # plot start and goal nodes in red and green
    mlab.points3d(start[0], start[1], start[2], color=(1, 0, 0), scale_factor=.2, resolution=60)
    mlab.points3d(goal[0], goal[1], goal[2], color=(0, 1, 0), scale_factor=.2, resolution=60)

    tree = rrt.tree
    for node, parent in tree.items():
        node = np.array(eval(node))
        # plot the nodes and connections between the nodes and their parents
        mlab.points3d(node[0], node[1], node[2], color=(0, 0, 1), scale_factor=.1)
        mlab.points3d(parent[0], parent[1], parent[2], color=(0, 0, 1), scale_factor=.1)
        mlab.plot3d([node[0], parent[0]], [node[1], parent[1]], [node[2], parent[2]], color=(0, 0, 0), tube_radius=0.01)


    # find the path from the start node to the goal node
    path = rrt.get_path()

    # Plot the paths
    mlab.plot3d(path[:, 0], path[:, 1], path[:, 2], color=(1, 1, 0), tube_radius=0.05)

    mlab.show()



