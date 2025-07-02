############ PREDEFINED HELPER FUNCTIONS: DO NOT EDIT ##############

import plotly.graph_objects as go
import numpy as np
import scipy.constants as sc
import math
import re

def read_xyz(input_conf):
    """
    Reads an XYZ file with multiple frames and extracts particle coordinates.

    Args:
        input_conf (str): Path to the XYZ file.

    Returns:
        numpy.ndarray: Coordinates of particles with shape (frames, particles, 3).
    """
    with open(input_conf, 'r') as xyz_file:
        n_particles = int(xyz_file.readline())
        xyz_file.readline()  # Skip the second line (box size or comment)

        all_frames = []
        while True:
            frame_coords = []
            for _ in range(n_particles):
                line = xyz_file.readline()
                if not line:
                    break
                frame_coords.append([float(x) for x in line.split()[1:4]])
            if not frame_coords:
                break
            all_frames.append(frame_coords)

            try:
                n_particles = int(xyz_file.readline())
                xyz_file.readline()  # Skip the second line
            except ValueError:
                break

    return np.array(all_frames)

def read_lammps_trj(lammps_trj_file):
    """
    Reads a LAMMPS trajectory dump file and extracts positions, velocities, forces, and box sizes.

    Args:
        lammps_trj_file (str): Path to the LAMMPS trajectory dump file.

    Returns:
        tuple: Positions, velocities, forces, and box sizes as numpy arrays.
    """
    def read_lammps_frame(trj):
        trj.readline()  # ITEM: TIMESTEP
        trj.readline()  # Skip timestep
        trj.readline()  # ITEM: NUMBER OF ATOMS
        n_atoms = int(trj.readline())
        trj.readline()  # ITEM: BOX BOUNDS
        box_bounds = np.array([list(map(float, trj.readline().split())) for _ in range(3)])
        trj.readline()  # ITEM: ATOMS id type x y z [vx vy vz fx fy fz]

        xyz, vxyz, fxyz = np.zeros((n_atoms, 3)), np.zeros((n_atoms, 3)), np.zeros((n_atoms, 3))
        for _ in range(n_atoms):
            line = trj.readline().split()
            atom_id = int(line[0]) - 1
            xyz[atom_id] = list(map(float, line[2:5]))
            if len(line) >= 8:
                vxyz[atom_id] = list(map(float, line[5:8]))
            if len(line) >= 11:
                fxyz[atom_id] = list(map(float, line[8:11]))

        return xyz, vxyz, fxyz, box_bounds

    positions, velocities, forces, box_sizes = [], [], [], []
    with open(lammps_trj_file, 'r') as f:
        while True:
            try:
                xyz, vxyz, fxyz, box = read_lammps_frame(f)
                positions.append(xyz)
                velocities.append(vxyz)
                forces.append(fxyz)
                box_sizes.append(box)
            except Exception:
                break

    return np.array(positions), np.array(velocities), np.array(forces), np.array(box_sizes)

def read_lammps_data(data_file, verbose=False):
    """
    Reads a LAMMPS data file
        Atoms
        Velocities
    Returns:
        lmp_data (dict):
            'xyz': xyz (numpy.ndarray)
            'vel': vxyz (numpy.ndarray)
        box (numpy.ndarray): box dimensions
    """
    print("Reading '" + data_file + "'")
    with open(data_file, 'r') as f:
        data_lines = f.readlines()

    # TODO: improve robustness of xlo regex
    directives = re.compile(r"""
        ((?P<n_atoms>\s*\d+\s+atoms)
        |
        (?P<box>.+xlo)
        |
        (?P<Atoms>\s*Atoms)
        |
        (?P<Velocities>\s*Velocities))
        """, re.VERBOSE)

    i = 0
    while i < len(data_lines):
        match = directives.match(data_lines[i])
        if match:
            if verbose:
                print(match.groups())

            elif match.group('n_atoms'):
                fields = data_lines.pop(i).split()
                n_atoms = int(fields[0])
                xyz = np.empty(shape=(n_atoms, 3))
                vxyz = np.empty(shape=(n_atoms, 3))

            elif match.group('box'):
                dims = np.zeros(shape=(3, 2))
                for j in range(3):
                    fields = [float(x) for x in data_lines.pop(i).split()[:2]]
                    dims[j, 0] = fields[0]
                    dims[j, 1] = fields[1]
                L = dims[:, 1] - dims[:, 0]

            elif match.group('Atoms'):
                if verbose:
                    print('Parsing Atoms...')
                data_lines.pop(i)
                data_lines.pop(i)

                while i < len(data_lines) and data_lines[i].strip():
                    fields = data_lines.pop(i).split()
                    a_id = int(fields[0])
                    xyz[a_id - 1] = np.array([float(fields[2]),
                                         float(fields[3]),
                                         float(fields[4])])

            elif match.group('Velocities'):
                if verbose:
                    print('Parsing Velocities...')
                data_lines.pop(i)
                data_lines.pop(i)

                while i < len(data_lines) and data_lines[i].strip():
                    fields = data_lines.pop(i).split()
                    va_id = int(fields[0])
                    vxyz[va_id - 1] = np.array([float(fields[1]),
                                         float(fields[2]),
                                         float(fields[3])])

            else:
                i += 1
        else:
            i += 1

    return xyz, vxyz, L

def write_frame(coords, L, vels, forces, trajectory_name, step):
    """
    Writes a single frame to a LAMMPS trajectory file.

    Args:
        coords (numpy.ndarray): Particle coordinates.
        L (float): Box size.
        vels (numpy.ndarray): Particle velocities.
        forces (numpy.ndarray): Particle forces.
        trajectory_name (str): Output trajectory file name.
        step (int): Current timestep.
    """
    n_particles = coords.shape[0]
    with open(trajectory_name, 'a') as file:
        file.write(f"ITEM: TIMESTEP\n{step}\n")
        file.write(f"ITEM: NUMBER OF ATOMS\n{n_particles}\n")
        file.write("ITEM: BOX BOUNDS pp pp pp\n")
        for _ in range(3):
            file.write(f"{-L/2:.6f} {L/2:.6f}\n")
        file.write("ITEM: ATOMS id type xu yu zu vx vy vz fx fy fz\n")

        for i in range(n_particles):
            file.write(f"{i+1} 1 {coords[i, 0]:.4f} {coords[i, 1]:.4f} {coords[i, 2]:.4f} "
                       f"{vels[i, 0]:.6f} {vels[i, 1]:.6f} {vels[i, 2]:.6f} "
                       f"{forces[i, 0]:.4f} {forces[i, 1]:.4f} {forces[i, 2]:.4f}\n")

def rdf(xyz, LxLyLz, n_bins=100, r_range=(0.01, 10.0)):
    """
    Computes the radial distribution function (RDF) for a set of particle coordinates.

    Args:
        xyz (numpy.ndarray): Particle coordinates per frame.
        LxLyLz (numpy.ndarray): Box dimensions.
        n_bins (int): Number of bins for the RDF histogram.
        r_range (tuple): Range of distances for the RDF.

    Returns:
        tuple: Radial distances and RDF values.
    """
    g_r = np.zeros(n_bins)
    rho = 0
    edges = np.linspace(r_range[0], r_range[1], n_bins + 1)

    for frame in xyz:
        for i, coord in enumerate(frame):
            d = np.abs(coord - frame)
            d = np.where(d > 0.5 * LxLyLz, LxLyLz - d, d)
            distances = np.sqrt(np.sum(d**2, axis=-1))
            distances = distances[distances > 0]
            g_r += np.histogram(distances, bins=edges)[0]

        rho += len(frame) / np.prod(LxLyLz)

    r = 0.5 * (edges[1:] + edges[:-1])
    shell_volumes = 4/3 * np.pi * (edges[1:]**3 - edges[:-1]**3)
    g_r /= rho * shell_volumes * len(xyz)

    return r, g_r

def view_trajectory(positions, box_size=None, speed_modes=None):
    """
    Efficiently plot 3D animation of particles using Plotly.

    Parameters:
        positions: numpy array of shape (n_frames, n_particles, 3)
        box_size: numpy.ndarray, size of the simulation cube of shape (2, 2, 2) with min and max of each dimension
        speed_modes: dict, mapping speed names to frame durations (ms)
    """
    if speed_modes is None:
        speed_modes = {'Slow': 200, 'Normal': 100, 'Fast': 30}

    n_frames, n_particles, _ = positions.shape

    def get_box_lines(positions):
        if box_size is None:
            # Calculate the min and max for each axis (x, y, z)
            min_vals = np.min(positions, axis=(0, 1))  # Get min along each axis (x, y, z)
            max_vals = np.max(positions, axis=(0, 1))  # Get max along each axis (x, y, z)

            # Create box corners based on the min/max values
            corners = np.array([
                [min_vals[0], min_vals[1], min_vals[2]], [max_vals[0], min_vals[1], min_vals[2]],
                [max_vals[0], max_vals[1], min_vals[2]], [min_vals[0], max_vals[1], min_vals[2]],
                [min_vals[0], min_vals[1], max_vals[2]], [max_vals[0], min_vals[1], max_vals[2]],
                [max_vals[0], max_vals[1], max_vals[2]], [min_vals[0], max_vals[1], max_vals[2]],
            ])
        else:
            # Create a cube with the given box size
            corners = np.array([
                [box_size[0, 0], box_size[1, 0], box_size[2, 0]],
                [box_size[0, 1], box_size[1, 0], box_size[2, 0]],
                [box_size[0, 1], box_size[1, 1], box_size[2, 0]],
                [box_size[0, 0], box_size[1, 1], box_size[2, 0]],
                [box_size[0, 0], box_size[1, 0], box_size[2, 1]],
                [box_size[0, 1], box_size[1, 0], box_size[2, 1]],
                [box_size[0, 1], box_size[1, 1], box_size[2, 1]],
                [box_size[0, 0], box_size[1, 1], box_size[2, 1]]
            ])

        # Define the edges of the cube by connecting the corners
        edges = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face edges
            [4, 5], [5, 6], [6, 7], [7, 4],  # Top face edges
            [0, 4], [1, 5], [2, 6], [3, 7]   # Vertical edges
        ]
        
        # Prepare x, y, z coordinates for the lines
        x, y, z = [], [], []
        for s, e in edges:
            for idx in [s, e, None]:  # Add None to break between edges
                if idx is None:
                    x.append(None)
                    y.append(None)
                    z.append(None)
                else:
                    x.append(corners[idx][0])
                    y.append(corners[idx][1])
                    z.append(corners[idx][2])

        # Return a Scatter3d object for the box
        return go.Scatter3d(
            x=x, y=y, z=z, mode='lines',
            line=dict(color='gray', width=2),
            showlegend=False
        )

    # Helper: Create frames once per speed
    def create_frames(speed_label):
        return [go.Frame(
            data=[
                go.Scatter3d(
                    x=positions[i, :, 0],
                    y=positions[i, :, 1],
                    z=positions[i, :, 2],
                    mode='markers',
                    marker=dict(size=2, color='blue', opacity=0.8),
                    showlegend=False
                )
            ],
            name=f'{speed_label}_frame{i}'
        ) for i in range(n_frames)]

    # Build all frames (per speed)
    all_frames = []
    for label in speed_modes:
        all_frames.extend(create_frames(label))

    # Initial frame
    init_data = [
        go.Scatter3d(
            x=positions[0, :, 0],
            y=positions[0, :, 1],
            z=positions[0, :, 2],
            mode='markers',
            marker=dict(size=2, color='blue', opacity=0.8),
            showlegend=False
        ),
        get_box_lines(positions)
    ]

    # Scene setup
    axis_template = dict(
        showbackground=False, showgrid=False, zeroline=False,
        showticklabels=False,
    )

    # Layout
    layout = go.Layout(
        scene=dict(xaxis=axis_template, yaxis=axis_template, zaxis=axis_template),
        margin=dict(l=0, r=0, t=100, b=100),
        updatemenus=[
            dict(
                type='buttons',
                showactive=False,
                direction='left',
                x=0.5, xanchor='center', y=1.15, yanchor='top',
                buttons=[
                    dict(label='Play (Normal)', method='animate', args=[
                        [f'Normal_frame{i}' for i in range(n_frames)],
                        dict(frame=dict(duration=speed_modes['Normal'], redraw=True), mode='immediate')
                    ]),
                    dict(label='Pause', method='animate', args=[
                        [None],
                        dict(mode='immediate', frame=dict(duration=0, redraw=False))
                    ]),
                    dict(label='Slow Down', method='animate', args=[
                        [f'Slow_frame{i}' for i in range(n_frames)],
                        dict(frame=dict(duration=speed_modes['Slow'], redraw=True), mode='immediate')
                    ]),
                    dict(label='Fast Forward', method='animate', args=[
                        [f'Fast_frame{i}' for i in range(n_frames)],
                        dict(frame=dict(duration=speed_modes['Fast'], redraw=True), mode='immediate')
                    ])
                ]
            )
        ],
        sliders=[dict(
            steps=[
                dict(method='animate', args=[
                    [f'Normal_frame{i}'],
                    dict(mode='immediate', frame=dict(duration=0, redraw=True))
                ], label=str(i)) for i in range(n_frames)
            ],
            x=0.1, y=-0.1, xanchor='left', yanchor='top',
            currentvalue=dict(prefix="Frame: ", font=dict(size=12)),
            len=0.8
        )]
    )

    fig = go.Figure(data=init_data, layout=layout, frames=all_frames)
    fig.show()

############ PREDEFINED HELPER FUNCTIONS: DO NOT EDIT ##############

############ INSERT YOUR FUNCTIONS HERE ##############
def initGrid(L, nPart, dim):
    '''
    function to determine particle positions and box size for a specified number of particles and density

    Start at the empty array coords in which we will place the coordinates of the particles.
    Create an additional variable called L which shall give the box dimension (float).
    Make sure to shift the coordinates to have (0,0,0) as the center of the box.

    Fill the box with particles without overlap and correct density

    You can check the positioning of particles using a scatter plot
    import matplotlib.pyplot as plt
    plt.figure()
    plt.scatter(x, y)
    plt.show()

    :param L: box dimension [Ang]
    :param rho: density of particles [number per unit length]

    :return: coordinates centered around origin [Ang]
    :return: box size [Ang]
    '''
    if dim == 1:
        # 1D case
        n_axis = nPart
        spac   = L / n_axis

        coords = np.zeros((nPart, 1))
        for i in range(nPart):
            coords[i, 0] = i * spac

        # centering part by substracting half extent to the coordinates
        half_extent = (n_axis - 1) / 2 * spac
        coords_centered = coords - half_extent
        return coords_centered, L

    elif dim == 2:
        #2D case
        n_axis = int(math.ceil(nPart**(1/2))) 
        spac   = L / n_axis
        coords = np.zeros((nPart, 2))
        count  = 0

        for i in range(n_axis):
            for j in range(n_axis):
                if count >= nPart:
                    break
                coords[count, 0] = i * spac
                coords[count, 1] = j * spac
                count += 1
            if count >= nPart:
                break

        
        half_extent = (n_axis - 1) / 2 * spac
        coords_centered = coords - half_extent
        return coords_centered, L

    elif dim == 3:
        # In 3D
        n_axis = int(math.ceil(nPart**(1/3)))
        spac   = L / n_axis
        coords = np.zeros((nPart, 3))
        count  = 0

        for i in range(n_axis):
            for j in range(n_axis):
                for k in range(n_axis):
                    if count >= nPart:
                        break
                    coords[count, 0] = i * spac
                    coords[count, 1] = j * spac
                    coords[count, 2] = k * spac
                    count += 1
                if count >= nPart:
                    break
            if count >= nPart:
                break

        
        half_extent = (n_axis - 1) / 2 * spac #center by subtracting half extent to te coordinates
        coords_centered = coords - half_extent
        return coords_centered, L

    else:
        raise ValueError("dim must be 1, 2, or 3.")
    
def initVel(nPart, T, dim, m):
    ''' 
    Inputs: nPart, T in Kelvin, dim and m in g/mol
    output: velocity in (Å/fs)
    
    '''

   
    kB_md = sc.Boltzmann * sc.Avogadro /1000  # boltzmann in  kJ/(mol K)
    
    sigma = np.sqrt(kB_md * T / m) #calculate the standard deviation for the velocity distribution in m/s

    if dim == 1:
        # 1D case
        velocities = np.random.normal(0, sigma, nPart)
        

    elif dim == 2:
        # 2D case
        velocities = np.random.normal(0, sigma, (nPart, 2))
        

    elif dim == 3:
        # 3D case
        velocities = np.random.normal(0, sigma, (nPart, 3))
        

    else:
        raise ValueError("dim must be 1, 2, or 3.")
    
    v_cm = np.mean(velocities, axis=0)
    velocities = velocities - v_cm  # Remove center of mass velocity to avoid drift
    
    
    KE_actual = 0.5 * (m/1000) * np.mean(np.sum(velocities**2, axis=1)) # Calculate actual kinetic energy in kJ/mol
    KE_target = 0.5 * dim * kB_md * T # calculate target kinetic energy in kJ/mol
    scaling = np.sqrt(KE_target / KE_actual) #calculate scaling factor to match target kinetic energy
    velocities = velocities * scaling  # Scale velocities to match target kinetic energy

    velocities_MD = velocities *1e-5 #convert velocities in A/fs
    #for debugging purposes calculate final temperature
    T_final = ((m/1000) * np.mean(np.sum(velocities**2, axis=1))) / (dim * kB_md)
    print("Final temperature:", T_final," K")


    return velocities_MD

def distances (pos, Lbox):
    
    dr = np.zeros((pos.shape[0], pos.shape[0], pos.shape[1])) #initialize the distance array
    r = np.zeros((pos.shape[0], pos.shape[0])) #initialize the distance matrix
    dr =  pos[:, np.newaxis, :] - pos[np.newaxis, :, :] #calculating the distance between the particles in a vectorized form by using the broadcasting feature of numpy
    dr = dr - Lbox * np.round(dr / Lbox) #apply periodic boundary conditions
    r = np.sqrt(np.sum(dr**2,axis=2)) #module of the distance

    return dr, r
def LJ_forces (pos,Lbox, Rcut, epsilon, sigma):
    """
    Units: pos, Lbox, sigma, Rcut all in Angstroms; epsilon in kJ/mol.
    Returns F  in kJ/mol * Angstroms (perfect for m = g/mol and time units in fs).
    """
    
    dr = np.zeros((pos.shape[0], pos.shape[0], pos.shape[1])) #initialize the distance array
    r = np.zeros((pos.shape[0], pos.shape[0])) #initialize the distance matrix
    N = pos.shape[0]  # calculate number of particles
    dim = pos.shape[1]
    F=np.zeros(pos.shape)
    dr, r = distances(pos, Lbox)
    
    mask = (r < Rcut) & (r > 1e-14) & np.triu(np.ones((N, N), dtype=bool), k=1) #condition where the Lennard-Jones potential is calculated, added condition to avoid division by zero and to avoid double counting
    
    r_accepted = r[mask]
    
    F_scal = (24 * epsilon * (2 * sigma**12 / r_accepted**13 - sigma**6 / r_accepted**7)) # derivative of the Lennard-Jones potential
    
    F_vec = (F_scal / r_accepted)[:, None] * dr[mask] # python broadcasting function to get vector with shape: (N, N, dim)
    
    #loop to apply Newtons law
    pairs = np.array(np.where(mask)).T  #  get list of (i, j) index pairs for which the force is computed by using transpose

    for k, (i, j) in enumerate(pairs): #loop for getting F vector
        # Apply equal and opposite forces to particle i and j (using Newton's third law)
        F[i] = F[i]+ F_vec[k]
        F[j] = F[j]- F_vec[k]
    
    return F

def velocityVerlet(pos, vel, F, dt, mass, Lbox, Rcut, epsilon, sigma):
    ''' pos, Lbox, Rcut,Sigma in angstroms, vel in angstrom/fs, F in kJ/mol *angstroms, dt in fs, mass in g/mol,'''
    
  
    
    mass = mass / 1000 #convert it back in kg/mol
    
    pos = pos + vel *dt + 1/(2*mass) *F* 1000*(1e10/1e15)**2 *dt**2 #calculate new position in Angstrom
    
    pos = pos - Lbox * np.round(pos / Lbox) #apply periodic boundary condition
    
    vel_dthalf = vel + 0.5* F* 1000*(1e10/1e15)**2/mass *dt #updated velocity after half timestep
    
    F_new = LJ_forces(pos, Lbox, Rcut, epsilon, sigma) #calculate the new LJ force
    
    vel = vel_dthalf + 0.5 * (F_new)* 1000*(1e10/1e15)**2 /mass * dt # update the velocity
    
    return vel, pos, F_new

def kineticEnergy(vel, mass):
    '''Converts velocity from Ang/fs in m/s and mass from g/mol to kg/mol and gives kinetic energy back in kJ/mol'''
    conv = (1e-10 / 1e-15)**2  # conversion: (Angstroms/fs)^2 -> m^2/s^2
    mass = mass /1000 #convert mass in kg/mol
    KE_J_per_mol = 0.5 * mass * np.sum(vel**2 * conv)  # calculate kinetic energy in J/mol
    return KE_J_per_mol / 1000  # kJ/mol

def potentialEnergy(pos, Lbox, Rcut, epsilon, sigma):
    ''' pos, Lbox, Rcut and sigma in Angstroms and epsilon in kJ/mol'''
    dr = np.zeros((pos.shape[0], pos.shape[0], pos.shape[1])) #initialize the distance array
    r = np.zeros((pos.shape[0], pos.shape[0])) #initialize the distance matrix
    N = pos.shape[0]  # calculate number of particles
   
    dr, r = distances(pos, Lbox)
    
    mask = (r < Rcut) & (r > 1e-14) & np.triu(np.ones((N, N), dtype=bool), k=1) #condition where the Lennard-Jones potential is calculated, added condition to avoid division by zero and to avoid double counting
    
    r_accepted = r[mask]
    energy = 4 * epsilon * ((sigma / r_accepted) ** 12 - (sigma / r_accepted) ** 6)  # calculate the Lennard-Jones potential

    return energy.sum()  # return the total potential energy

def temperature(KE, N, dim): #calculate temperature from kinetic energy (formula from lecture 7 slide 26)
    '''Kinetic energy in kJ/mol, returns temp in K'''
    kB_md = sc.Boltzmann * sc.Avogadro / 1000 # kJ/(mol·K)
    return (2 * KE) / (N* dim * kB_md)  # calculate temperature from kinetic energy


def pressure(pos, vel, mass, Lbox, Rcut, epsilon, sigma):
    '''pos, Lbox, Rcut and sigma in angstroms, vel in ang/fs, epsilon in kJ/mol, mass in g/mol'''
    
    dr = np.zeros((pos.shape[0], pos.shape[0], pos.shape[1])) #initialize the distance array
    r = np.zeros((pos.shape[0], pos.shape[0])) #initialize the distance matrix
    N = pos.shape[0]  # number of particles
    V = Lbox ** 3
    
    kB_md = sc.Boltzmann * sc.Avogadro / 1000
    dim = pos.shape[1]  # number of dimensions
    dr, r = distances(pos, Lbox)  # calculate distances between particles
    mask = (r < Rcut) & (r > 1e-14) & np.triu(np.ones((N, N), dtype=bool), k=1) #condition where the Lennard-Jones potential is calculated, added condition to avoid division by zero and to avoid double counting
    
    r_accepted = r[mask]
    
    F_scal = (24 * epsilon * (2 * sigma**12 / r_accepted**13 - sigma**6 / r_accepted**7)) # derivative of the Lennard-Jones potential
    
    F_vec    = (F_scal / r_accepted)[:,None] * dr[mask] # python broadcasting function to get vector with shape: (N, N, dim)
    r_vec = dr[mask]  # extract corresponding distance vectors for the accepted pairs
    KE = kineticEnergy(vel, mass)  # calculate kinetic energy
    T = temperature(KE, N, dim) #calculate temperature from kinetic energy
    P = N * kB_md * T / V +  1/(6*V) * np.sum(np.einsum('ij,ij->i',F_vec, r_vec))  # calculate pressure using the virial theorem
    P_MPa = P * (1e27/sc.Avogadro) # convert pressure to MPa
   

    return P_MPa  # return pressure in MPa



def MDSolve (pos, vel, mass, Lbox, Rcut, epsilon, sigma,dt, steps, sample_freq, log_file, traj_file): #function to solve the molecular dynamics case
    step = 0  # initialize step counter
    N = pos.shape[0]  # number of particles
    dim = pos.shape[1]  # number of dimensions
    

        
    F = LJ_forces(pos, Lbox, Rcut, epsilon, sigma) #calculate the initial forces using the Lennard-Jones potential
 
 
    with open(log_file, 'w') as flog:  # open log file to write simulation data
        flog.write('Step    T(K)      KE(kJ)        PE(kJ)       TotE(kJ)\n')
        
    open(traj_file, 'w').close()    

    for step in range(steps):
            
            vel, pos, F_new = velocityVerlet(pos, vel, F, dt, mass, Lbox, Rcut, epsilon, sigma) #update positions and velocities using the velocity Verlet algorithm after first time step
            #pos = pos - Lbox * np.round(pos / Lbox)  # apply periodic boundary conditions
            F = F_new
           
           
            
            if step % sample_freq == 0:
                KE = kineticEnergy(vel, mass)  # calculate kinetic energy
                P = pressure(pos, vel, mass, Lbox, Rcut, epsilon, sigma)  # calculate pressure
                Upot = potentialEnergy(pos, Lbox, Rcut, epsilon, sigma)  # calculate potential energy
                Tempi = temperature(KE, pos.shape[0], pos.shape[1])  # calculate temperature
                print('%i %.12f %.12f %.12f %.12f' %(step, Tempi, KE, Upot, KE+Upot))
                with open(log_file, 'a') as flog:
                    flog.write(f"{step:6d} {Tempi:10.3f} {KE:12.5e} {Upot:12.5e} {Upot+KE:12.5e}\n") # write the data to the log file

                write_frame(pos, Lbox, vel, F_new, traj_file, step)  # write the current frame to the trajectory file
                                    
                
    return pos, vel, F  # return the final positions, velocities, and forces

#define function to read log lamps film to read properly the file
def read_log_lammps(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()

    data = []
    start_reading = False
    for line in lines:
        if "Step" in line and "Temp" in line:
            start_reading = True
            continue
        if start_reading:
            if line.strip() == '' or line.startswith("Loop"):
                break
            parts = line.split()
            if len(parts) == 6:
                data.append([float(x) for x in parts])
    
    return np.array(data)

def velocityVerletThermostat(pos, Lbox, vel,F,zeta,mass, dt,kbTemp,sigma,epsilon, Rcut, Q, nPart):

    ''' pos, Lbox, Rcut,Sigma in angstroms, vel in angstrom/fs, F in kJ/mol *angstroms, dt in fs, mass in g/mol'''
  
    
    
    KE = kineticEnergy(vel, mass)  # calculate kinetic energy
    mass_kg = mass / 1000  # convert mass to kg/mol for calculations
    pos = pos + vel *dt + 0.5 *dt**2 *(F* 1000*(1e10/1e15)**2/mass_kg-zeta*vel)
    zeta = zeta + 0.5* dt/Q * (KE-((3*nPart+1)/2) * kbTemp) #update the random force xi 
    vel = vel + 0.5* dt * (F* 1000*(1e10/1e15)**2/mass_kg - zeta*vel) #update the velocity 
    
    F = LJ_forces(pos, Lbox, Rcut, epsilon, sigma) # update the forces using the Lennard-Jones potential
    KE = kineticEnergy(vel, mass)  # recalculate kinetic energy after force update
    
    zeta = zeta + 0.5* dt/Q * (KE-((3*nPart+1)/2) * kbTemp) #update the random force xi again
    vel = (vel + 0.5*dt * (F* 1000*(1e10/1e15)**2)/mass_kg)/(1 + 0.5*dt *zeta) #update the velocity again
   

    return pos, vel, F, zeta  # return updated positions, velocities, forces and random force

def MDSolve_v2 (nPart,pos,T, vel, mass, Lbox, Rcut, epsilon, sigma,dt, steps, sample_freq,Q,zeta, log_file, traj_file): #function to solve the molecular dynamics case
    step = 0  # initialize step counter
    N = pos.shape[0]  # number of particles
    dim = pos.shape[1]  # number of dimensions
    kB_kJ = sc.Boltzmann * sc.Avogadro / 1000   #kJ/mol *K
    kbTemp = kB_kJ * T       #kJ /mol

        
    F = LJ_forces(pos, Lbox, Rcut, epsilon, sigma) #calculate the initial forces using the Lennard-Jones potential
 
 
    with open(log_file, 'w') as flog:  # open log file to write simulation data
        flog.write('Step    T(K)    KE(kJ)    PE(kJ)    TotE(kJ)    P(MPa)\n')
        
    open(traj_file, 'w').close()    

    for step in range(steps):
            
            pos,vel, F_new, zeta = velocityVerletThermostat(pos, Lbox, vel,F,zeta,mass, dt,kbTemp,sigma,epsilon, Rcut, Q, nPart) #update positions and velocities using the velocity Verlet algorithm after first time step
            pos = pos - Lbox * np.round(pos / Lbox)  # apply periodic boundary conditions
            F = F_new
           
           
            
            if step % sample_freq == 0:
                KE = kineticEnergy(vel, mass)  # calculate kinetic energy
                P = pressure(pos, vel, mass, Lbox, Rcut, epsilon, sigma)  # calculate pressure
                Upot = potentialEnergy(pos, Lbox, Rcut, epsilon, sigma)  # calculate potential energy
                Tempi = temperature(KE, pos.shape[0], pos.shape[1])  # calculate temperature
                print('%i %.12f %.12f %.12f %.12f' %(step, Tempi, KE, Upot, KE+Upot))
                with open(log_file, 'a') as flog:
                    flog.write(f"{step:6d} {Tempi:10.3f} {KE:12.5e} {Upot:12.5e} {Upot+KE:12.5e} {P:12.5e}\n") # write the data to the log file

                write_frame(pos, Lbox, vel, F_new, traj_file, step)  # write the current frame to the trajectory file
                                    
                
    return pos, vel, F  # return the final positions, velocities, and forces


############ INSERT YOUR FUNCTIONS HERE ##############