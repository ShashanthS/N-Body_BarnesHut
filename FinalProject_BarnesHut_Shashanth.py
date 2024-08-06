import matplotlib.pyplot as plt
import numpy as np
import time
from matplotlib.animation import FuncAnimation


# np.random.seed(1)  # For reproducibility


class QuadTreeNode:
    """
    A class used to represent the node of a quadtree.
    A node can represent an individual particle, or a section of the computational grid
    with mass equal to the total mass of all nested particles.

    Methods
    -------
    update_center_of_mass(self, position, mass)
        Update the total mass and COM of the node with the given position and mass.

    insert(self, position, mass)
        Insert a body with the given position and mass. If node is already occupied,
        then subdivide into four children, each representing a quadrant within the boundary
        box of the node.

    compute_forces(self, particle, theta, G, eps)
        Compute the forces on the given particle from all the nested particles in the node.
    """

    def __init__(self, x_min, x_max, y_min, y_max):
        """
        Initializes the class
        """

        self.bbox = [x_min, x_max, y_min, y_max]
        self.children = None
        self.total_mass = None
        self.center_of_mass = np.zeros(2)

    def update_center_of_mass(self, position, mass):
        """
        Updates com of a node given a new position and mass
        """
        COM_new = (self.center_of_mass * self.total_mass + position * mass) / (self.total_mass + mass)
        self.center_of_mass = COM_new

    def insert(self, position, mass):
        """
        Inserts a particle (postion, mass) into the node. Checks for the three different cases
         - un-initalized
         - Initialzied but no children
         - Has children
        """

        if self.total_mass is None:
            # Node is unoccupied: update total mass and COM
            self.total_mass = mass
            self.center_of_mass = position

        elif self.children is None:
            # Node is occupied, but not yet subdivided: subdivide and distribute old and new particle

            # Check for equality - this happens if the two particles share a location
            if (self.center_of_mass == position).all():

                # If the two particles share mass, update just the mass of the node
                # This treats the particles as a single particle with a mass of the sum of the two
                self.total_mass += mass
                # print("The particle being inserted has the same position as another particle")

            else:
                self.children = [None, None, None, None]

                # Figuring out which quadrant the already present particle would be in
                # This is done by using the mass of the previous particle, given by total mass and the position of the
                # particle given by the COM position. Create a node at that point

                quad_index, bbox_new = get_quadrant(self.bbox, self.center_of_mass)
                self.children[quad_index] = QuadTreeNode(bbox_new[0], bbox_new[1], bbox_new[2], bbox_new[3])
                self.children[quad_index].insert(self.center_of_mass, self.total_mass)

                # Now insert the new particle again:
                self.insert(position, mass)

        else:
            # We come here if there are sub-nodes already present
            # We need to figure out which quadrant the particle goes into and insert the particle there
            quad_index, bbox_new = get_quadrant(self.bbox, position)

            # If there is already a node present for the quadrant, insert into that node
            if self.children[quad_index] is not None:
                self.children[quad_index].insert(position, mass)

            # If no node is present, create the node and insert into node
            else:
                self.children[quad_index] = QuadTreeNode(bbox_new[0], bbox_new[1], bbox_new[2], bbox_new[3])
                self.children[quad_index].insert(position, mass)

            # Update the total mass of the current node - only needs to be updated here, not the prev if statement
            self.update_center_of_mass(position, mass)
            self.total_mass += mass

    def compute_accel(self, particle, theta, G=1, eps=0.00000001):
        """
         Calculates and returns
         the acceleration on a particle from the nested particles in the node.
         The code checks if:
         - the node is uninitialized
         - the particle is the node itself
         - whether to use the total mass of the node, or a direct summation
           of the nested particles.

        particle is a numpy array of the form [x, y, v_x, v_y, mass]

        """

        # Position (x,y) are the first two elements of the particle array
        pos = particle[:2]
        particle_mass = particle[-1]

        # First, check if it has sub-nodes.
        if self.children is None:

            # If it does not have sub-nodes, check whether the node is the particle itself.
            if not (self.center_of_mass == pos).all():

                # If not, calculate force
                a_temp = (G * self.total_mass *
                          ((self.center_of_mass - pos) / ((np.linalg.norm(self.center_of_mass - pos) ** 3) + eps)))
                # print('I am at a particle')
                return a_temp

            # If the node is the particle itself, the Force is 0
            else:
                return np.array([0.0, 0.0])

        else:
            # Check if we can just use the mass and COM of the node, based on theta
            d = np.linalg.norm(self.center_of_mass - pos)
            # print(np.abs(self.bbox[1] - self.bbox[0]) / d)
            if np.abs(self.bbox[1] - self.bbox[0]) / d < theta:
                a = G * (self.total_mass * (self.center_of_mass - pos) /
                         ((np.linalg.norm(self.center_of_mass - pos) ** 3) + eps))
                # print('Calculating Force at a Node')
                return a
            else:
                a = np.array([0.0, 0.0])
                for child in self.children:
                    if child is not None:
                        a += child.compute_accel(particle, theta, G, eps)

                return a

    def plot(self, plot_bbox=True, plot_particles=True):
        """
        Method to plot the bbox and particles within a given node.
        """
        if plot_bbox:
            # First, plot the bbox
            width = 0.3

            # Uncomment below to mark the intersection points as well
            # plt.plot(self.bbox[0], self.bbox[2], 'k.', markersize=3)
            # plt.plot(self.bbox[1], self.bbox[2], 'k.', markersize=3)
            # plt.plot(self.bbox[0], self.bbox[3], 'k.', markersize=3)
            # plt.plot(self.bbox[1], self.bbox[3], 'k.', markersize=3)

            plt.plot([self.bbox[0], self.bbox[1]], [self.bbox[2], self.bbox[2]], 'k', linewidth=width)
            plt.plot([self.bbox[0], self.bbox[0]], [self.bbox[2], self.bbox[3]], 'k', linewidth=width)
            plt.plot([self.bbox[0], self.bbox[1]], [self.bbox[3], self.bbox[3]], 'k', linewidth=width)
            plt.plot([self.bbox[1], self.bbox[1]], [self.bbox[2], self.bbox[3]], 'k', linewidth=width)
        if True:
            # If no sub-nodes, then we only have one particle, we can then plot the particle as well
            if self.children is None:
                if plot_particles:
                    plt.plot(self.center_of_mass[0], self.center_of_mass[1], 'r.', markersize=3)
            # Loop through sub-nodes, if present, and plot the bbox and particles recursively
            else:
                for child in self.children:
                    if child is not None:
                        child.plot(plot_bbox=plot_bbox, plot_particles=plot_particles)


class QuadTree:
    """
    A class that represents the full tree, which contains all the nodes
    and particles.

    Attributes
    ----------
    particles : array
        An array containing the particles in the simulation. Each particle should have a
        position, velocity, and mass attribute.
    theta : float
        The resolution parameter of the Barnes-Hut algorithm.
    root : QuadTreeNode
        The root node of the tree.

    Methods
    -------
    update_center_of_mass(self, position, mass)
        Update the total mass and COM of the node with the given position and mass.

    insert(self)
        Insert and distribute all the particles into the tree.

    """

    def __init__(self, root, particles, theta):
        """
        Initialize the root node and set the particles and theta attributes
        """
        self.root = root
        self.particles = particles
        self.theta = theta

    def insert(self):
        """
        Insert the particles one by one into the root of the tree
        """
        for this_particle in self.particles:
            self.root.insert(this_particle[:2], this_particle[-1])


def get_quadrant(bbox, position):
    """
    This function finds which quadrant the given position belongs to within bbox. It returns the index of this quadrant
    along with the boundary of the quadrant.

    bbox is a list/array of the form [x_min, x_max, y_min, y_max]
    position is a numpy array of the form [x,y]

    returns: quad_index, bbox
    - quad_index: int; index of the quadrant to which the given position belongs
    - bbox: numpy array of the bbox to which the position belongs
    return none if position not within bbox
    """

    x_mid = (bbox[0] + bbox[1]) / 2
    y_mid = (bbox[2] + bbox[3]) / 2

    # Ensure that particle is actually within the bbox - return none if not
    if position[0] < bbox[0] or position[0] > bbox[1] or position[1] < bbox[2] or position[1] > bbox[3]:
        print(position, bbox)
        print('Particle does not lie within this boundary box')
        return

    # Figure out which quadrant the particle is in

    elif position[0] <= x_mid:
        if position[1] <= y_mid:
            # Goes to bottom left quadrant
            return 2, np.array([bbox[0], x_mid, bbox[2], y_mid])
        elif position[1] > y_mid:
            # Goes to top left quadrant
            return 0, np.array([bbox[0], x_mid, y_mid, bbox[3]])

    elif position[0] > x_mid:
        if position[1] <= y_mid:
            # Goes to bottom right quadrant
            return 3, np.array([x_mid, bbox[1], bbox[2], y_mid])
        elif position[1] > y_mid:
            # Goes to top right quadrant
            return 1, np.array([x_mid, bbox[1], y_mid, bbox[3]])


def initialize_particles(N, r_V=1, M=1.0, G=1.0, config_vel='zero_vel', config_m="uniform", central_mass=False):
    """
    write a function that initialized N particles with
    the initial conditions.
    The distribution of the positions follows the plummer model

    The velocity distribution can be specified as one of
     - 'normal_dist', 'zero_vel', 'rotational' or 'differential'

    The mass distribution is either 'uniform' or 'normal'. It is normalized such that the
    total mass always adds up to 1. If the parameter M is less than one, we assume a central particle
    of mass 1-M to be present.

    returns
    particles: numpy array of the form [pos_x, pos_y, v_x, v_y, mass]
    """

    # Generating positions according to the Plummer model

    # Sample a density first
    p = np.random.uniform(0, 1, N)
    r = r_V / np.sqrt(p ** (-2 / 3) - 1)

    # Sample a random position on circle of radius r
    p_0 = np.random.uniform(0, 1, N)
    p_1 = np.random.uniform(0, 1, N)
    x = (r * np.sin(np.arccos(2 * p_0 - 1)) * np.cos(2 * np.pi * p_1))
    y = (r * np.sin(np.arccos(2 * p_0 - 1)) * np.sin(2 * np.pi * p_1))

    # Generate masses:

    if config_m == 'uniform':
        masses = np.ones(N) * M / N

    elif config_m == 'normal':
        # Create normal distribution of masses
        masses = np.abs(np.random.normal(0, .5, N))

        # Normalize to ensure total mass is M
        masses = M * masses / np.sum(masses)

    else:
        print("Specified mass config does not exist")
        return

    # Generating random velocities

    if config_vel == 'normal_dist':
        v = np.sqrt(2) / 2
        v_x = v * np.random.normal(0, 1, N)
        v_y = v * np.random.normal(0, 1, N)

    # Zero Initial Velocity
    elif config_vel == 'zero_vel':
        v_x = np.zeros(N)
        v_y = np.zeros(N)

    elif config_vel == 'rotational':
        theta = np.arctan2(y, x)
        v = np.sqrt(2) / 2

        v_x = v * (-np.sin(theta)) * np.random.normal(0, 1, N)
        v_y = v * (np.cos(theta)) * np.random.normal(0, 1, N)

    elif config_vel.lower() == 'differential':
        # We need to assign velocities based on location of particle
        v = np.zeros(N)
        for i, rad in enumerate(r):
            M_tot = np.sum(masses[rad < r]) + (1 - M)
            v[i] = np.sqrt(G * M_tot / rad)
            # print(v[i], rad, M)

        # Figure out x and y comps of the velocities
        theta = np.arctan2(y, x)
        v_x = v * (-np.sin(theta))
        v_y = v * (np.cos(theta))

    else:
        print('Velocity config is not defined')
        return

    particles = np.column_stack((x, y, v_x, v_y, masses))

    return particles


def add_individual_particles(current_particles, new_particle):
    new_particle_array = np.append(current_particles, [new_particle], axis=0)
    return new_particle_array


def integrate_euler(tree, tau=0.01, T=10):
    history = [tree.particles]
    t = 0
    temp_count = 0
    while t < T:

        # For the last time step
        if t + tau > T:
            tau = T - t

        # We want to calculate the force on each particle for a given quadtree
        forces = np.zeros(shape=(len(tree.particles), 2))
        for i, particle in enumerate(tree.particles):
            forces[i] = tree.root.compute_accel(particle, tree.theta)

        # Do a time step
        new_particles = tree.particles.copy()
        # Update Position
        new_particles[:, 0:2] = tree.particles[:, 0:2] + tree.particles[:, 2:4] * tau

        # Update Velocity
        new_particles[:, 2:4] = tree.particles[:, 2:4] + forces * tau

        # Update tree with new particles:
        max_val = np.ceil(np.max(np.abs(new_particles[:, 0:2])))
        root_node = QuadTreeNode(-max_val, max_val, -max_val, max_val)
        tree = QuadTree(root_node, new_particles, tree.theta)
        tree.insert()

        history.append(new_particles)
        t += tau
        temp_count += 1
    plt.show()
    return np.array(history)


def integrate_rk4(mytree, tau=0.001, T=10):
    """
    There are bugs in this, please ignore and do not use
    """
    tree = mytree
    history = []
    t = 0
    # T = 5
    # print('Tree Particles\n', tree.particles)
    while t <= T:
        # print('Time: ', t)
        # Calculate forces
        forces = np.zeros(shape=(len(tree.particles), 2))

        for i, particle in enumerate(tree.particles):
            forces[i] = tree.root.compute_accel(particle, tree.theta)
            # print(forces[i])
        # print('Accels\n', forces * .25)
        # Define k1 as the derivatives at current location
        k1 = np.append(tree.particles[:, 2:4], forces, axis=1)

        temp_particle = tree.particles.copy()

        # Follow k1 to the halfway point using an Euler step
        temp_particle[:, :4] = tree.particles[:, :4] * tau / 2 * k1

        # We need to create a tree at this timestep to calculate forces
        max_val = np.ceil(np.max(np.abs(temp_particle[:, 0:2])))
        temp_root_node = QuadTreeNode(-max_val, max_val, -max_val, max_val)

        temp_tree = QuadTree(temp_root_node, temp_particle, tree.theta)
        temp_tree.insert()

        # Compute forces at halfway point
        for i, particle in enumerate(temp_particle):
            forces[i] = temp_tree.root.compute_accel(particle, temp_tree.theta)

        # Define k2 as the derivatives obtained at the halfway point
        k2 = np.append(temp_particle[:, 2:4], forces, axis=1)

        # Follow k2 to the halfway point
        temp_particle = tree.particles.copy()
        temp_particle[:, :4] = tree.particles[:, :4] * tau / 2 * k2

        # We need to create a tree at this timestep to calculate forces
        max_val = np.ceil(np.max(np.abs(temp_particle[:, 0:2])))
        temp_root_node = QuadTreeNode(-max_val, max_val, -max_val, max_val)

        temp_tree = QuadTree(temp_root_node, temp_particle, tree.theta)
        temp_tree.insert()

        # Compute forces at halfway point
        for i, particle in enumerate(temp_particle):
            forces[i] = temp_tree.root.compute_accel(particle, temp_tree.theta)

        k3 = np.append(temp_particle[:, 2:4], forces, axis=1)

        # Now, we follow k3 through the full time step
        temp_particle = tree.particles.copy()

        temp_particle[:, :4] = tree.particles[:, :4] * tau * k3

        # We need to create a tree at this timestep to calculate forces
        max_val = np.ceil(np.max(np.abs(temp_particle[:, 0:2])))
        temp_root_node = QuadTreeNode(-max_val, max_val, -max_val, max_val)

        temp_tree = QuadTree(temp_root_node, temp_particle, tree.theta)
        temp_tree.insert()

        # Compute forces at halfway point
        for i, particle in enumerate(temp_particle):
            forces[i] = temp_tree.root.compute_accel(particle, temp_tree.theta)

        # Define k4 as the derivatives at this point
        k4 = np.append(temp_particle[:, 2:4], forces, axis=1)

        # Define k as the weighted average over k1, k2, k3, k4
        k = 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        new_particles = tree.particles.copy()
        # print('New Particles\n', new_particles)
        new_particles[:, :4] = new_particles[:, :4] + k * tau
        # print('New Particles\n', new_particles)
        #         print('New Particles\n', new_particles)

        # Update tree with new particles:
        max_val = np.ceil(np.max(np.abs(new_particles[:, 0:2])))
        root_node = QuadTreeNode(-max_val, max_val, -max_val, max_val)

        tree = QuadTree(root_node, new_particles, tree.theta)
        tree.insert()

        history.append(new_particles)
        t += tau

    history = np.array(history)
    # print(k1, k2)
    # print('History\n', history)
    # print(temp_particle)
    return history


def integrate_velverlet(mytree, tau=.01, T=10):
    tree = mytree
    t = 0
    hist = []  # Store particles in a list
    while t < T:

        # For the last time step
        if t + tau > T:
            tau = T - t

        # Calculate forces
        forces = np.zeros(shape=(len(tree.particles), 2))
        for i, particle in enumerate(tree.particles):
            forces[i] = tree.root.compute_accel(particle, tree.theta)

        # We want to take half steps
        v_half = tree.particles[:, 2:4] + tau / 2 * forces
        r_next = tree.particles[:, 0:2] + tau * v_half

        # Now, we need to calculate the forces at the new position
        # Define temp particles to create tree - note that velocities don't matter here
        half_particles = np.append(r_next, tree.particles[:, 2:5], axis=1)
        bbox_max = np.ceil(np.max(np.abs(r_next)))

        root_node = QuadTreeNode(-bbox_max, bbox_max, -bbox_max, bbox_max)
        tree_half = QuadTree(root_node, half_particles, tree.theta)
        tree_half.insert()

        forces_half = np.zeros(shape=(len(tree.particles), 2))
        for i, particle in enumerate(tree_half.particles):
            forces_half[i] = tree_half.root.compute_accel(particle, tree_half.theta)

        v_next = v_half + tau / 2 * forces_half
        new_particles = np.append(r_next, v_next, axis=1)
        new_particles = np.append(new_particles, np.array([tree.particles[:, -1]]).reshape(len(tree.particles), 1),
                                  axis=1)

        root_node = QuadTreeNode(-bbox_max, bbox_max, -bbox_max, bbox_max)

        tree = QuadTree(root_node, new_particles, tree.theta)
        tree.insert()

        hist.append(new_particles)
        t += tau
    return np.array(hist)


def display(hist, tau=0.01, fname="TestFigs/Test.gif", make_anim=True, make_plot=False):
    """
    Display function to create an animation of particles

    hist is an array contain arrays of particles corresponding to each instant of time in
    the simluation
    """
    if make_anim:
        fig, ax = plt.subplots()
        line, = ax.plot([], [], '.', markersize=3)
        ax.set(xlim=[-10, 10], ylim=[-10, 10])
        ax.set_aspect('equal')
        ax.set_xlabel('x (N-body units)')
        ax.set_ylabel('y (N-body units)')
        ax.grid()

        text = ax.text(0.8, 0.8, '', transform=ax.transAxes)

        def animate(frame):
            x = hist[frame, :, 0]
            y = hist[frame, :, 1]

            line.set_xdata(x)
            line.set_ydata(y)

            text.set_text(f't = {tau * frame:.3}')

            return line

            # ax.plot(x, y, '.')
            # plt.show()

        anim = FuncAnimation(fig=fig, func=animate, frames=len(hist), interval=1)
        anim.save(filename=fname, fps=60, dpi=300)
        plt.show()

    if make_plot:
        fig, ax = plt.subplots()
        ax.set(xlim=[-10, 10], ylim=[-10, 10])
        ax.set_aspect('equal')
        ax.set_xlabel('x (N-body units)')
        ax.set_ylabel('y (N-body units)')
        ax.grid()

        ax.plot(hist[:, :, 0], hist[:, :, 1], linestyle='dotted', alpha=0.5)
        ax.set_aspect('equal')
        fig.savefig(f'{fname[:-4]}_hist.png')
        plt.show()


def calculate_energy(hist, tau=0.01, G=1, plot=True, fname='TempName.png'):
    """
    Calculates the total energy in a system given hist

    hist is an array of particles corresponding to each instant of time in the simulation
    """

    time = np.linspace(0, tau * len(hist), len(hist))
    E_time = []
    E_gp_list = []
    E_KE_list = []

    # Iterate through sets of particles at different time instants
    for particles_t in hist:

        # Iterate through all particles in given time instant
        E_gp = 0
        for i, current_part in enumerate(particles_t):
            other_particles = particles_t[i + 1:]

            E_gp -= np.sum((G * current_part[-1] * other_particles[:, -1]
                            / np.linalg.norm(other_particles[:, 0:2] - current_part[0:2])))

        E_KE = np.sum(.5 * particles_t[:, -1] * (particles_t[:, 2] ** 2 + particles_t[:, 3] ** 2))
        # print(E_KE, E_gp)
        E = E_KE + E_gp

        E_gp_list.append(E_gp)
        E_KE_list.append(E_KE)
        E_time.append(E)

    E_time = np.array(E_time)
    E_KE_list = np.array(E_KE_list)
    E_gp_list = np.array(E_gp_list)

    print("Calculated Energy, Generating Plot")

    if plot:
        fig, ax = plt.subplots()
        ax.plot(time, E_KE_list / E_KE_list[0], color='green', label='Kinetic Energy')
        ax.plot(time, E_time / E_time[0], linestyle='dotted', color='black', label='Total energy')
        ax.plot(time, E_gp_list / E_gp_list[0], color='tab:blue', label='Potential Energy')
        ax.set_xlabel('Time (N-body units)')
        ax.set_ylabel('Normalized Total Energy (N-body units)')
        fig.savefig(f'{fname[:-4]}_energy.png', dpi=600)

    return E_time


# - - - - - - - - - - - - - - - - - - - - - - - - - - -
# - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Testing + Running
# - - - - - - - - - - - - - - - - - - - - - - - - - - -
# - - - - - - - - - - - - - - - - - - - - - - - - - - -


def wrapper_sanitycheck():
    particle_1 = np.array([-0.2, 0.3, 0, 0, 0.25])
    particle_2 = np.array([-0.1, 0.15, 0, 0, 0.25])
    particle_3 = np.array([0.25, 0.25, 0, 0, 0.25])
    particle_4 = np.array([0.25, -0.25, 0, 0, 0.25])

    init_particles = np.array([particle_1, particle_2, particle_3, particle_4])

    root_node = QuadTreeNode(-0.5, 0.5, -0.5, 0.5)
    tree = QuadTree(root_node, init_particles, 0)
    tree.insert()

    a = []
    for particle in init_particles:
        a.append(tree.root.compute_accel(particle, tree.theta))

    print(np.array(a) * 0.25)


def wrapper_genenralrun():
    """
    Wrapper to run the entire set up.
    Saves the generated animations and figures to specified directory
    """
    n_particles = 20
    tau = 0.01
    T = 2
    theta = 0.6
    M_particles = 0.2

    # Filename to store data
    filename = f'Results/Set 9/velverlet_{n_particles}particles_{T}T_{M_particles}M_{theta}theta.mp4'
    filename = 'Testing.mp4'

    random_parts = initialize_particles(n_particles, M=M_particles, config_vel='differential', config_m='normal')

    # Add a mass externally
    large_mass = np.array([[-7, -7, 4, 4, 1-M_particles]])
    random_parts = np.append(random_parts, large_mass, axis=0)

    # Figure out root bbox using max value of position, create root QuadTreeNode
    max_val = np.ceil(np.max(np.abs(random_parts[:, 0:2])))
    root_node = QuadTreeNode(-max_val, max_val, -max_val, max_val)

    # Create QuadTree from root node
    temp = QuadTree(root_node, random_parts, theta)
    temp.insert()
    temp.root.plot(plot_bbox=True, plot_particles=True)
    plt.gca().set_aspect('equal')
    plt.savefig(f'{filename[:-4]}_grid.png', dpi=600)

    print('Integrating')

    hist = integrate_velverlet(mytree=temp, tau=tau, T=T)
    np.save(f'{filename[:-4]}_hist', hist)

    print('Saved history, creating animation')

    display(hist, tau=0.01, fname=filename, make_plot=True)
    print('Animation Created, calculating energy')

    E_t = calculate_energy(hist, fname=filename)
    np.save(f'{filename[:-4]}_E', E_t)
    print('Saved energy data')


def wrapper_twobody():
    large_mass = np.array([[0, 0, 0, 0, 0.9]])
    small_mass = np.array([1, 0, 0, np.sqrt(0.9)])
    random_parts = np.append(large_mass, small_mass, axis=0)
    # print(random_parts)

    # Figure out root bbox using max value of position, create root QuadTreeNode
    max_val = np.ceil(np.max(np.abs(random_parts[:, 0:2])))
    root_node = QuadTreeNode(-max_val, max_val, -max_val, max_val)

    # Create QuadTree from root node
    theta = 0.0
    temp = QuadTree(root_node, random_parts, theta)
    temp.insert()
    temp.root.plot(plot_bbox=True, plot_particles=True)
    plt.gca().set_aspect('equal')
    plt.show()

    hist = integrate_rk4(tree=temp, tau=0.01, T=10)

    print('Creating Animation')
    display(hist, fname='TestFigs/2body_1.mp4')


def wrapper_force_theta_err():
    """
    Checking the error due to various theta values
    """

    n_avg = 10
    n_particles = 200
    thetas = np.linspace(0.1, 3, 20)
    errors_tot = np.zeros(shape=(len(thetas)))
    for i in range(n_avg):
        # Create distribution of particles

        init_particles = initialize_particles(n_particles, 1, config_m='normal', config_vel='differential')

        # Figure out root bbox using max value of position, create root QuadTreeNode
        max_val = np.ceil(np.max(np.abs(init_particles[:, 0:2])))
        root_node = QuadTreeNode(-max_val, max_val, -max_val, max_val)

        # Run with theta=0 as a base for comparison
        theta = 0.0
        tree = QuadTree(root_node, init_particles, theta)
        tree.insert()

        forces_0 = np.zeros(shape=(len(tree.particles), 2))
        for i, particle in enumerate(tree.particles):
            forces_0[i] = tree.root.compute_accel(particle, tree.theta)

        # Run for different values of thetas

        # hists = []
        forces_list = []
        for theta in thetas:
            tree.theta = theta
            forces = np.zeros(shape=(len(tree.particles), 2))
            for i, particle in enumerate(tree.particles):
                forces[i] = tree.root.compute_accel(particle, tree.theta)
            forces_list.append(forces)

        errors = []
        for forces in forces_list:
            errors.append(np.sqrt(np.sum(((forces - forces_0) / forces_0) ** 2) / np.sqrt(len(forces))))
        errors_tot += np.array(errors)

    # Plotting

    plt.plot(thetas, errors_tot / n_avg)
    plt.yscale('log')
    # plt.xscale('log')
    plt.axhline(0, linestyle='dotted', color='grey')
    plt.xlabel('Theta')
    plt.ylabel('Error')
    plt.title('Error for varying values of theta')
    plt.savefig(f'Results/Errors/Errors_wrt_theta_{n_particles}particles.png', dpi=600)
    plt.show()

    return


def wrapper_timing():
    """
    Used to determine performance of the algorithm
    """
    # We want to create grids of N particles and compute forces once for each
    N_vals = [300, 400, 600, 1000, 1200, 1500]
    times = []
    for N in N_vals:
        # Generate N particles
        t_avg = 0
        n = 20
        for i in range(n):
            init_particles = initialize_particles(N, config_vel='rotational', config_m='normal')

            # Figure out root bbox using max value of position, create root QuadTreeNode
            max_val = np.ceil(np.max(np.abs(init_particles[:, 0:2])))
            root_node = QuadTreeNode(-max_val, max_val, -max_val, max_val)

            # Create tree with theta
            theta = 1

            tree = QuadTree(root_node, init_particles, theta)
            tree.insert()

            start = time.perf_counter()
            for particle in tree.particles:
                tree.root.compute_accel(particle, theta)

            t_avg += time.perf_counter() - start
        times.append(t_avg / n)
    plt.plot(N_vals, times, '.')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('N')
    plt.ylabel('Time to run (s)')
    plt.title(f'Performance for theta of {theta}')
    N_vals = np.array(N_vals)
    plt.plot(N_vals, times[2] / (N_vals[2] * np.log(N_vals[2])) * (np.array(N_vals) * np.log(N_vals)),
             label='NlogN', linestyle='dashed')
    plt.plot(N_vals, times[2] / N_vals[2] ** 2 * N_vals ** 2, label='N^2',
             linestyle='dashed')
    plt.legend()
    plt.savefig(f'Results/Performance Test/perf_{theta}theta.png')
    plt.show()
    # return


# Run the Sanity Check
# wrapper_sanitycheck()

# Run General Function - Modify within the wrapper function
wrapper_genenralrun()

# Compare accuracy of algorithm for different values of theta
# wrapper_force_theta_err()

# T
# wrapper_timing()

print('All Done! :)')

def UI():
    """
    Required Parameters:
    - Integrator to be used
    - Time Step (tau)
    - Total Run Time (T)
    -
    :return:
    """
    return
