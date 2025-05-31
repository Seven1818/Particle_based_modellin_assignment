############ PREDEFINED HELPER FUNCTIONS: DO NOT EDIT ##############

import plotly.graph_objects as go
import numpy as np
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

# def initGrid():
#     returns initial configuration

############ INSERT YOUR FUNCTIONS HERE ##############