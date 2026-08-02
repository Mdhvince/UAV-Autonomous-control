import ast
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
        assert self.space_limits_lw[2] <= self.start[2] <= self.space_limits_up[2], \
            "The z location of the start must be within the z space limits"
        assert self.space_limits_lw[2] <= self.goal[2] <= self.space_limits_up[2], \
            "The z location of the goal must be within the z space limits"

    def run(self):
        old_cost = np.inf

        for it in range(self.max_iterations):

            new_node = self._generate_random_node()
            nearest_node = self._find_nearest_node(new_node)
            new_node = self._adapt_random_node_position(new_node, nearest_node)
            neighbors = self._find_valid_neighbors(new_node)

            if len(neighbors) == 0: continue

            best_neighbor = self._find_best_neighbor(neighbors, new_node)
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

    @staticmethod
    def _node_key(node: np.ndarray) -> str:
        return str(np.round(node, 2).tolist())

    def _cost_to_come(self, node: np.ndarray) -> float:
        """
        Cost of the path from the start to `node` following the tree edges.
        """
        cost = 0.0
        current = node
        while not np.array_equal(current, self.start):
            parent = self.tree[RRTStar._node_key(current)]
            cost += np.linalg.norm(np.asarray(current, dtype=float) - np.asarray(parent, dtype=float))
            current = parent
        return cost

    def _find_best_neighbor(self, neighbors, new_node):
        """
        Find the neighbor that yields the cheapest path from the start to the new node, where the cost of a
        neighbor is its cost-to-come through the tree plus the length of the edge to the new node.
        """
        costs = []
        for neighbor in neighbors:
            cost = self._cost_to_come(neighbor) + np.linalg.norm(neighbor - new_node)
            costs.append(cost)

        best_neighbor = neighbors[np.argmin(costs)]
        return best_neighbor

    def _update_tree(self, node, new_node):
        """
        Link the new node to the tree with `node` as parent, unless the new node is already in the tree
        with a cheaper cost-to-come.
        """
        node_key = RRTStar._node_key(new_node)
        node_parent = np.round(node, 2)

        if np.array_equal(node_parent, new_node):
            return

        if node_key in self.tree:
            current_cost = self._cost_to_come(new_node)
            candidate_cost = self._cost_to_come(node_parent) + np.linalg.norm(new_node - node_parent)
            if current_cost <= candidate_cost:
                return

        self.all_nodes.append(new_node)
        self.tree[node_key] = node_parent

    def _rewire_safely(self, neighbors, new_node):
        """
        For every neighbor (except the parent of the new node and the start), re-wire it to the new node
        if passing through the new node is cheaper than its current path.
        """
        has_rewired = False
        new_node_cost = self._cost_to_come(new_node)

        for neighbor in neighbors:
            if np.array_equal(neighbor, self.start):
                continue

            if np.array_equal(neighbor, self.tree[RRTStar._node_key(new_node)]):
                # if the neighbor is already the parent of the new node, skip
                continue

            current_cost = self._cost_to_come(neighbor)
            cost_through_new_node = new_node_cost + np.linalg.norm(neighbor - new_node)

            if cost_through_new_node < current_cost:
                self.tree[RRTStar._node_key(neighbor)] = np.round(new_node, 2)
                has_rewired = True

        return has_rewired

    def _is_valid_connection(self, node, new_node):
        """
        Check if the connection between the candidate node and the new node (random or goal) is collision-free
        """
        if self.obstacles is None:
            return True

        for obstacle in self.obstacles:
            if RRTStar._segment_intersects_cuboid(node, new_node, obstacle):
                return False

        return True

    @staticmethod
    def _segment_intersects_cuboid(node1: np.ndarray, node2: np.ndarray, cuboid: np.ndarray) -> bool:
        """
        Exact segment vs axis-aligned cuboid intersection test (slab method).
        :param node1: segment start point [x, y, z]
        :param node2: segment end point [x, y, z]
        :param cuboid: bounds [x_min, x_max, y_min, y_max, z_min, z_max]
        :return: True if the segment intersects the cuboid
        """
        direction = np.asarray(node2, dtype=float) - np.asarray(node1, dtype=float)
        t_min, t_max = 0.0, 1.0

        for axis in range(3):
            low, high = cuboid[2 * axis], cuboid[2 * axis + 1]
            if abs(direction[axis]) < 1e-12:
                if node1[axis] < low or node1[axis] > high:
                    return False
                continue

            t_low = (low - node1[axis]) / direction[axis]
            t_high = (high - node1[axis]) / direction[axis]
            if t_low > t_high:
                t_low, t_high = t_high, t_low

            t_min = max(t_min, t_low)
            t_max = min(t_max, t_high)
            if t_min > t_max:
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
                raise Exception("A problem occurred while computing the path, please restart the algorithm")

        cost = RRTStar.path_cost(path)
        return np.array(path[::-1]).reshape(-1, 3), cost


if __name__ == "__main__":

    # NED frame: z points down, so flying 7 m above ground means z = -7
    start = np.array([0, 0, 0])
    goal = np.array([7, 7, -7])

    space_limits = np.array([[0., 0., -10.], [10., 10., 0.]])

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
        node = np.array(ast.literal_eval(node))
        fig.add_trace(go.Scatter3d(x=[node[0], parent[0]], y=[node[1], parent[1]], z=[node[2], parent[2]], mode='lines', line=dict(width=1, color='blue')))

    # find the path from the start node to the goal node
    path = rrt.best_path

    # Plot the paths
    fig.add_trace(go.Scatter3d(x=path[:, 0], y=path[:, 1], z=path[:, 2], mode='lines', line=dict(width=5, color='yellow')))

    fig.show()
