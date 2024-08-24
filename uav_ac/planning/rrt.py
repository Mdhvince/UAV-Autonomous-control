import copy
import time
import numpy as np
import plotly.graph_objects as go


class RRTStar:
    """
    Rapidly-exploring Random Tree (RRT*) algorithm
    """

    def __init__(self, space_limits, start, goal, max_distance, max_iterations, obstacles=None):
        self.space_limits_lw, self.space_limits_up = space_limits[0], space_limits[1]
        self.start = np.round(start, 2)
        self.goal = np.round(goal, 2)
        self.step_size = max_distance
        self.max_iterations = max_iterations
        self.obstacles = obstacles
        self.epsilon = 0.15

        self.neighborhood_radius = 1.5 * max_distance
        self.all_nodes = [self.start]

        self.tree = {}
        self.best_path = None
        self.best_tree = None

        self.dynamic_it_counter = 0
        self.dynamic_break_at = self.max_iterations / 10

        assert self.neighborhood_radius > self.step_size, "Neighborhood radius must be larger than step size"
        assert self.space_limits_up[2] > self.start[2], "Upper limit on z must be > than the z location of the start"
        assert self.space_limits_up[2] > self.goal[2], "Upper limit on z must be > than the z location of the goal"

    def run(self):
        old_cost = np.inf

        for it in range(self.max_iterations):

            new_node = self._generate_random_node()
            nearest_node = self._find_nearest_node(new_node)
            new_node = self._adapt_random_node_position(new_node, nearest_node)
            neighbors = self._find_valid_neighbors(new_node)

            if len(neighbors) == 0: continue

            best_neighbor = self._find_best_neighbor(neighbors)
            self._update_tree(best_neighbor, new_node)
            has_rewired = self._rewire_safely(neighbors, new_node)

            if self._is_path_found(self.tree):
                path, cost = self.get_path(self.tree)

                if has_rewired and cost > old_cost:  # sanity check
                    raise Exception("Cost increased after rewiring")

                if cost < old_cost:
                    print("Iteration: {} | Cost: {}".format(it, cost))
                    self.store_best_tree()
                    old_cost = cost
                    self.dynamic_it_counter = 0
                else:
                    self.dynamic_it_counter += 1
                    print(
                        "\r Percentage to stop unless better path is found: {}%".format(
                            np.round(self.dynamic_it_counter / self.dynamic_break_at * 100, 2)), end="\t")

                if self.dynamic_it_counter >= self.dynamic_break_at:
                    break

        if not self._is_path_found(self.best_tree):
            raise Exception("No path found")

        self.best_path, cost = self.get_path(self.best_tree)
        print("\nBest path found with cost: {}".format(cost))

    def store_best_tree(self):
        """
        Update the best tree with the current tree if the cost is lower
        """
        # deepcopy is very important here, otherwise it is just a reference. copy is enough for the
        # dictionary, but not for the numpy arrays (values of the dictionary) because they are mutable.
        self.best_tree = copy.deepcopy(self.tree)

    @staticmethod
    def path_cost(path):
        """
        Calculate the cost of the path
        """
        cost = 0
        for i in range(len(path) - 1):
            cost += np.linalg.norm(path[i + 1] - path[i])
        return cost

    def _generate_random_node(self):
        # with probability epsilon, sample the goal
        if np.random.uniform(0, 1) < self.epsilon:
            return self.goal

        x_rand = np.random.uniform(self.space_limits_lw[0], self.space_limits_up[0])
        y_rand = np.random.uniform(self.space_limits_lw[1], self.space_limits_up[1])
        z_rand = np.random.uniform(self.space_limits_lw[2], self.space_limits_up[2])
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

                # cost to arrive to the neighbor
                current_cost = np.linalg.norm(neighbor - self.start)

                # cost to arrive to the neighbor through the new node
                potential_new_cost = np.linalg.norm(new_node - self.start) + np.linalg.norm(neighbor - new_node)

                if potential_new_cost < current_cost:
                    # if it is cheaper to arrive to the neighbor through the new node, re-wire (update the parent of the
                    # neighbor to the new node)
                    self.tree[str(np.round(neighbor, 2).tolist())] = new_node
                    return True
        return False

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

    def _is_path_found(self, tree):
        """
        Check if the goal node is in the tree as a child of another node
        """
        goal_node_key = str(np.round(self.goal, 2).tolist())
        return goal_node_key in tree.keys()

    def get_path(self, tree):
        """
        Get the path from the goal node to the start node and compute its cost
        """

        path = [self.goal]
        node = self.goal

        s_time = time.time()

        while not np.array_equal(node, self.start):
            node = tree[str(np.round(node, 2).tolist())]
            path.append(node)

            if time.time() - s_time > 5:
                # restart
                print("Restarting...")
                self.run()
                # raise Exception("A problem occurred while computing the path.")

        cost = RRTStar.path_cost(path)
        return np.array(path[::-1]).reshape(-1, 3), cost


if __name__ == "__main__":

    start = np.array([0, 0, 0])
    goal = np.array([7, 7, 7])

    space_limits = np.array([[0., 0., 0.9], [10., 10., 10.]])

    rrt = RRTStar(
        space_limits,
        start=start,
        goal=goal,
        max_distance=3,
        max_iterations=1000,
        obstacles=None,
    )
    rrt.run()

    fig = go.Figure()

    # plot start and goal nodes in red and green
    fig.add_trace(go.Scatter3d(x=[start[0]], y=[start[1]], z=[start[2]], mode='markers', marker=dict(size=5, color='red')))
    fig.add_trace(go.Scatter3d(x=[goal[0]], y=[goal[1]], z=[goal[2]], mode='markers', marker=dict(size=5, color='green')))

    tree = rrt.best_tree
    for node, parent in tree.items():
        node = np.array(eval(node))
        fig.add_trace(go.Scatter3d(x=[node[0], parent[0]], y=[node[1], parent[1]], z=[node[2], parent[2]], mode='lines', line=dict(width=1, color='blue')))

    # find the path from the start node to the goal node
    path = rrt.best_path

    # Plot the paths
    fig.add_trace(go.Scatter3d(x=path[:, 0], y=path[:, 1], z=path[:, 2], mode='lines', line=dict(width=5, color='yellow')))

    fig.show()
