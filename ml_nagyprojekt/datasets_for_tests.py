import numpy as np


def make_two_blobs(n=500, p=500, blob1_mean=-2, blob1_sd=0.5, blob2_mean=2, blob2_sd=0.5):
    """Creates two Gaussian blobs with equal numbers of points

    Optional inputs:
    n ~ Total number of data points (half of which will go to each blob)
    p ~ Number of columns (dimensionality) in blob data
    blob1_mean ~ Mean of normal distribution from which blob1 points are drawn
    blob1_sd ~ Standard deviation of normal distribution from which blob1 points are drawn
    blob2_mean ~ Mean of normal distribution from which blob2 points are drawn
    blob2_sd ~ Standard deviation of normal distribution from which blob2 points are drawn

    Outputs:
    X ~ n x dim array of blob data
    labs ~ Labels corresponding to the two different blobs"""

    X = np.r_[np.random.normal(blob1_mean, blob1_sd, [n // 2, p]), np.random.normal(blob2_mean, blob2_sd, [n // 2, p])]
    labs = np.repeat(np.array([1, 2]), n // 2)
    return X, labs


def make_line(x=np.linspace(start=0, stop=50, num=200), slope=2, intercept=0, noise=3):
    """Creates single line of points with random scatter

    Optional inputs:
    x ~ Numpy array of x values over which the line should be generated
    slope ~ Slope of line
    intercept ~ Y-intercept of line
    noise ~ Standard deviation of random noise

    Outputs:
    X ~ 2 column array with x values in first column and y (line) values in second column"""

    return np.c_[x, slope * x + intercept + np.random.normal(loc=0, scale=3, size=len(x))]


def make_parallel_lines(x=np.linspace(start=0, stop=50, num=200), num_lines=2, spacing=17,
                        slope=2, intercept=0, noise=3):
    """Creates parallel lines of points with random scatter

    Optional inputs:
    x ~ Numpy array of x values over which the line should be generated
    num_lines ~ Number of parallel lines
    spacing ~ Amount of space between each parallel line
    slope ~ Slope of lines
    intercept ~ Y-intercept of first line plotted
    noise ~ Standard deviation of random noise

    Outputs:
    X ~ (2 * len(x)) x (num_lines + 1) array with x values in first column and y (line) values in second column
    labs ~ Labels corresponding to the two different parallel lines"""

    # Set original line
    X = make_line(x, slope, intercept, noise)

    # Add additional lines until num_lines is reached
    for i in range(1, num_lines):
        X = np.r_[X, make_line(x, slope, intercept + i * spacing, noise)]

    # Generate labels
    labs = np.repeat(np.arange(1, num_lines + 1), len(x))

    # Return results
    return X, labs


###################################################################################
# Creating more complex, 3-dimensional shapes.
###################################################################################

def make_two_3d_circles(radius=1, theta=np.linspace(0, 2 * np.pi, 201)):
    """Generate data and labels for 3D linked circles of specified radius. Circles are rotated 90-degrees
    relative to each other and are linked such that one passes through the exact center of the other

    Optional inputs:
    radius ~ Radius of the two circles
    theta ~ Numpy array corresponding to theta values of interest (which, by default, create full circles)

    Outputs:
    X ~ (2 * len(theta) x 3) array with coordinates of circles
    labs ~ Labels corresponding to the two different circles"""

    # Generate X and labs
    x = np.r_[np.zeros(len(theta)), radius * np.cos(theta)]
    y = np.r_[radius * np.cos(theta), np.zeros(len(theta))]
    z = np.r_[radius * np.sin(theta), radius * np.sin(theta) + radius]
    X = np.c_[x, y, z]
    labs = np.repeat(np.array([1, 2]), len(theta))

    # Return results
    return X, labs


def make_trefoil_knot(size=1, phi=np.linspace(0, 2 * np.pi, 300)):
    """Generate data and labels for 3D tiefoil knot

    Optional inputs:
    size ~ Scaling factor to increase / decrease size of tiefoil knot
    phi ~ Numpy array corresponding to phi values of interest (which, by default, create full tiefoil knot)

    Outputs:
    X ~ (len(theta)) x 3) array with coordinates for tiefoil know
    labs ~ Color labels for tiefoil knot (simply a color gradient, given that there are not distinct classes)"""

    # Generate X and labs
    X = size * np.c_[np.sin(phi) + 2 * np.sin(2 * phi), \
                     np.cos(phi) - 2 * np.cos(2 * phi), \
        -np.sin(3 * phi)]
    labs = np.repeat(np.array([1, 2, 3]), len(phi) / 3)

    # Return results
    return X, labs
