import numpy as np
import streamlit as st
from matplotlib import pyplot as plt

from planning.minimum_snap import MinimumSnap
from planning.rrt import RRTStar



@st.cache_data
def run_rrt():
    start = np.array([0, 0, 0])
    goal = np.array([7, 7, 7])
    space_limits = np.array([[0., 0., 0.9], [10., 10., 10.]])

    rrt = RRTStar(space_limits, start=start, goal=goal, max_distance=1, max_iterations=1000, obstacles=None)
    rrt.run()

    tree = rrt.best_tree
    path = rrt.best_path
    min_snap = MinimumSnap(path, None, 1.0, 0.01)
    trajectory = min_snap.get_trajectory()
    return rrt, tree, path, trajectory


def plot_path(rrt, tree, path, trajectory=None):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(rrt.start[0], rrt.start[1], rrt.start[2], marker="o", color="r", s=100, label="start")
    ax.scatter(rrt.goal[0], rrt.goal[1], rrt.goal[2], marker="o", color="g", s=100, label="goal")

    for node, parent in tree.items():
        node = np.array(eval(node))
        ax.scatter(node[0], node[1], node[2], marker=".", color="k", s=10)
        if parent is not None:
            ax.scatter(parent[0], parent[1], parent[2], marker=".", color="k", s=10)
        ax.plot([node[0], parent[0]], [node[1], parent[1]], [node[2], parent[2]], color="k", linewidth=0.5)
    ax.plot(path[:, 0], path[:, 1], path[:, 2], color="y", linewidth=2, label="Path")

    if trajectory is not None:
        ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], color="b", linewidth=2, label="Optimized trajectory")

    # remove axis
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.legend()
    return fig



st.set_page_config(page_title="RRT*", layout="wide", initial_sidebar_state="expanded")
st.markdown("### RRT Star with Minimum Snap Trajectory Optimization")
st.markdown("")

col1, col2 = st.columns(2)

with st.spinner():
    rrt, tree, path, trajectory = run_rrt()

with col1:
    inner_col1, inner_col2 = st.columns(2)
    rrt_button = inner_col1.button("RRT")
    snap_button = inner_col2.button("Minimum Snap")

if rrt_button:
    with st.spinner("Plotting RRT path"):
        fig = plot_path(rrt, tree, path)
        col1.pyplot(fig)

if snap_button:
    with st.spinner("Plotting Minimum Snap Trajectory"):
        fig = plot_path(rrt, tree, path, trajectory=trajectory)
        col1.pyplot(fig)

with col2.expander("RRT Tree [node: parent]"):
    st.write(tree)

