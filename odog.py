"""
The odog.py module contains the ODOG and Region classes needed to build and run
the one dimension, one group diffusion equation solver.
"""

# Import libraries
import numpy as np 

class ODOG:

	def __init__(self, geom=None):
		"""Initialize the ODOG class.

		Initialization involves the creation of a regions attribute, which
		is an empty list. This is a container for Region objects that 
		create the geometry.

		Parameters
		----------
		geom : str
			The type of geometry for the solver. Currently supported options
			are 1D: spherical, slab, or inf. cylindrical.
		"""
		
		# Initialize empty list of regions for the geometry
		self.regions = []

		# Set the geometry attribute, if it is supported, as well as the alpha
		# value corresponding to the geometry type
		if geom.lower() not in ['slab', 'cylindrical', 'spherical']:
			raise Exception('Provide a supported 1D geometry!')
		else:
			# Set geometry attribute
			self.geom = geom
			# Set alpha
			if geom.lower() == 'slab':
				self.alpha = 0
			elif geom.lower() == 'cylindrical':
				self.alpha = 1
			elif geom.lower() == 'spherical': 
				self.alpha = 2

	def add_region(self, start=0, end=1, D=0, siga=0, nusigf=0, nnodes=0):
		"""Method to add a region to the geometry.

		This method creates a region given the start and end positions, 
		defined by absorption and fission macroscopic cross sections, 
		a diffusion coefficient, nu, and a number of mesh nodes.

		Parameters 
		----------
		start : float 
			Start position of the region, in units of cm.
		end : float
			End position of the region, in units of cm.
		D : float
			Region diffusion coefficient, in units of cm.
		siga : float
			Region macroscopic absorption cross section, in units of cm^-1.
		nusigf : float
			Region macroscopic fission cross section multiplied by fissile
			material nu, in units of cm^-1. 
		nnodes : int
			The number of mesh nodes in the region.
		"""

		# Create the region using the Region class and append it to the 
		# list of regions for the geometry.
		self.regions.append(Region(start, end, D, siga, nusigf, nnodes))

	def solve(self, eps=1e-3):
		"""Method to solve the diffusion equatio problem.

		Parameters
		----------
		eps : float
			Convergence tolerance on multiplication factor k between power iterations.
		"""

		# Set up the problem
		self.setup()

		# Build the matrices
		self.build_matrices()

		# Set initial guesses for the flux and the multiplication factor
		self.phi = np.ones((self.N, 1))
		self.k = 1

		# Compute the source vector and psi vector
		self.S = np.matmul(self.F, self.phi)
		self.psi = (1/self.k)*self.S

		# Initialize convergence flag
		converged = False 

		# Compute the eigenvalue and the flux
		while not converged:

			# Compute the new flux
			self.phi = np.matmul(np.linalg.inv(self.A), self.psi)

			# Compute the new source
			self.S = np.matmul(self.F, self.phi)

			# Set the previous eigenvalue attribute and compute the new eigenvalue
			self.prev_k = self.k
			self.k = (np.matmul(np.transpose(self.S), self.S))/(np.matmul(np.transpose(self.S), self.psi))

			# Compute the new psi
			self.psi = (1/self.k)*self.S

			# Check convergence criteria
			tau = abs((self.k - self.prev_k)/self.k)
			if tau < eps:
				# Set convergence flag to true
				converged = True
				# Add 0 to end of flux because this method assumes the flux at the outer
				# boundary is 0.
				self.phi = np.append(self.phi, np.array([0]))

	def setup(self):
		"""Method to setup the problem.

		This method sets up the full system geometric and material discretization
		by aggregating the attributes of each region.
		"""

		# Compound the region attributes to solver geometry attributes
		for i, region in enumerate(self.regions):
			if i == 0:
				self.r = region.r 
				self.deltar = region.deltar
				self.halfr = region.halfr
				self.D = np.array([region.D]*len(region.halfr))
				self.siga = np.array([region.siga]*len(region.halfr))
				self.nusigf = np.array([region.nusigf]*len(region.halfr))
			else:
				self.r = np.append(self.r, region.r[1:])
				self.deltar = np.append(self.deltar, region.deltar)
				self.halfr = np.append(self.halfr, region.halfr)
				self.D = np.append(self.D, np.array([region.D]*len(region.halfr)))
				self.siga = np.append(self.siga, np.array([region.siga]*len(region.halfr)))
				self.nusigf = np.append(self.nusigf, np.array([region.nusigf]*len(region.halfr)))

	def build_matrices(self):
		"""Method to build the matrices used in the diffusion equation solution.

		This method uses the full problem aggregated geometric and material attributes to
		build the A and F matrices used in the linear algebra solution of the diffusion 
		equation.
		"""

		# Preallocate matrices of zeros
		self.N = len(self.r)  - 1
		self.A = np.zeros((self.N, self.N))
		self.F = np.zeros((self.N, self.N))

		# NOTE: Lengths of self.r and (self.halfr, self.deltar, etc...) are NOT THE SAME
		# THEREFORE THE INDICES HAVE DIFFERENT MEANINGS IN EACH

		for j in range(0, self.N):
			# Inner boundary cell edge
			if j == 0:
				aj = ((self.halfr[j]**self.alpha)*self.D[j])/self.deltar[j] 
				bj = ((self.halfr[j]**self.alpha)*self.D[j])/self.deltar[j] + \
				     ((self.r[j]**self.alpha)/2)*(self.siga[j]*self.deltar[j])
				cj = 0
				Fj = (self.r[j]**self.alpha)*(self.nusigf[j]*(self.deltar[j]/2))
			else:
				aj = ((self.halfr[j]**self.alpha)*self.D[j])/self.deltar[j] 
				bj = ((self.halfr[j-1]**self.alpha)*self.D[j-1])/self.deltar[j-1] + \
				     ((self.halfr[j]**self.alpha)*self.D[j])/self.deltar[j] + \
				     ((self.r[j]**self.alpha)/2)*(self.siga[j-1]*self.deltar[j-1] + self.siga[j]*self.deltar[j])
				cj = ((self.halfr[j-1]**self.alpha)*self.D[j-1])/self.deltar[j-1]
				Fj = (self.r[j]**self.alpha)*(self.nusigf[j-1]*(self.deltar[j-1]/2) + self.nusigf[j]*(self.deltar[j]/2))

			# Set components in matrices
			if j == 0:
				self.A[j][j] = bj 
				self.A[j][j+1] = -1*aj
			elif j == self.N-1:
				self.A[j][j] = bj
				self.A[j][j-1] = -1*cj
			else:
				self.A[j][j] = bj
				self.A[j][j+1] = -1*aj
				self.A[j][j-1] = -1*cj

			self.F[j][j] = Fj

class Region:

	def __init__(self, start, end, D, siga, nusigf, nnodes):
		"""Initialize the Region class.

		Initialization sets the characterizing attributes of the region.

		Parameters 
		----------
		start : float 
			Start position of the region, in units of cm.
		end : float
			End position of the region, in units of cm.
		D : float
			Region diffusion coefficient, in units of cm.
		siga : float
			Region macroscopic absorption cross section, in units of cm^-1.
		nusigf : float
			Region macroscopic fission cross section multiplied by fissile
			material nu, in units of cm^-1. 
		nnodes : int
			The number of mesh nodes in the region.		
		"""

		# Set the attributes of the region
		self.start = start
		self.end = end 
		self.D = D 
		self.siga = siga
		self.nusigf = nusigf 
		self.r = np.linspace(start, end, nnodes)
		self.deltar = self.r[1:] - self.r[0:-1]
		self.halfr = self.r[1:]-(self.deltar/2)