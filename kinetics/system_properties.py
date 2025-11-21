import numpy as np
from dataclasses import dataclass, field
from typing import List, Iterable, Union

@dataclass
class Particle:
    position: np.ndarray
    velocity: np.ndarray
    mass: float

    def __post_init__(self):
        # ensure numpy arrays and correct shapes
        self.position = np.asarray(self.position, dtype=float)
        self.velocity = np.asarray(self.velocity, dtype=float)
        if self.position.shape != self.velocity.shape:
            raise ValueError("position and velocity must have the same shape")
        if self.position.ndim != 1:
            self.position = self.position.ravel()
            self.velocity = self.velocity.ravel()
        self.mass = float(self.mass)
        if self.mass <= 0:
            raise ValueError("mass must be positive")

    @classmethod
    def from_tuple(cls, tup):
        """Create a Particle from (r, r_dot, m) tuple."""
        r, r_dot, m = tup
        return cls(np.asarray(r, dtype=float), np.asarray(r_dot, dtype=float), float(m))

    def as_tuple(self):
        """Return (position, velocity, mass) tuple."""
        return (self.position, self.velocity, self.mass)


@dataclass
class System:
    particles: List[Particle] = field(default_factory=list)

    # Physical state attributes (initialized in __post_init__)
    total_mass: float = field(init=False)
    cm_position: np.ndarray = field(init=False)
    cm_velocity: np.ndarray = field(init=False)
    linear_momentum_vec: np.ndarray = field(init=False)
    translational_energy: float = field(init=False)
    rotational_energy: float = field(init=False)

    def __post_init__(self):
        # initialize physical attributes to sensible defaults
        self.total_mass = 0.0
        self.cm_position = np.zeros(0, dtype=float)
        self.cm_velocity = np.zeros(0, dtype=float)
        self.linear_momentum_vec = np.zeros(0, dtype=float)
        self.angular_momentum_vec = np.zeros(0, dtype=float)
        self.translational_energy = 0.0
        self.rotational_energy = 0.0

    def add_particle(self, particle: Particle) -> None:
        """Add a single Particle to the system."""
        if not isinstance(particle, Particle):
            raise TypeError("particle must be a Particle instance")
        self.particles.append(particle)

    def add_particles(self, particles: Iterable[Particle]) -> None:
        """Add multiple Particle instances to the system."""
        for p in particles:
            self.add_particle(p)

    def __iadd__(self, other: Union[Particle, Iterable[Particle]]):
        """Support in-place addition: system += particle or system += [p1, p2]."""
        if isinstance(other, Particle):
            self.add_particle(other)
        else:
            self.add_particles(other)
        return self

    def _resolve_input(self, particles: Union[None, Iterable[Particle]]) -> List[Particle]:
        """Helper: return a list of Particle objects (default to self.particles)."""
        if particles is None:
            return list(self.particles)
        return [p if isinstance(p, Particle) else Particle.from_tuple(p) for p in particles]

    # def compute_ang_mom(system: List[Particle], r: tuple = (np.zeros(0, dtype=float), np.zeros(0, dtype=float))) -> np.ndarray:
    #     """
    #     Compute total angular momentum of the system (Nms) at the specified point r (pos, vel). If r is not specified, computes at CoM.
    #     If 'particles' is None, use particles stored in the System.
    #     Updates self.angular_momentum_vec (and mass/CM attributes).
    #     Returns: numpy array vector of momentum.
    #     """
    #     angular_momentum_vec = sum(p.mass * np.cross((p.position - r[0]), (p.velocity - r[1])) for p in system)

    #     return angular_momentum_vec

    def _update_mass_and_cm(self, system: List[Particle]) -> None:
        """Update all physical quantities impacted by system mass layout"""
        if len(system) == 0:
            self.total_mass = 0.0
            self.cm_position = np.zeros(0, dtype=float)
            self.cm_velocity = np.zeros(0, dtype=float)
            self.linear_momentum_vec = np.zeros(0, dtype=float)
            self.angular_momentum_vec = np.zeros(0, dtype=float)
            return

        self.total_mass = sum(p.mass for p in system)
        self.cm_position = sum(p.mass * p.position for p in system) / self.total_mass
        self.cm_velocity = sum(p.mass * p.velocity for p in system) / self.total_mass
        self.linear_momentum_vec = self.total_mass * self.cm_velocity
        # self.angular_momentum_vec = self.compute_ang_mom((self.cm_position, self.cm_velocity))

    def kinetic_energy(self, particles: Iterable[Particle] = None) -> tuple:
        """
        Compute translational and rotational (deformation) kinetic energy.
        If 'particles' is None, use particles stored in the System.
        Updates self.translational_energy and self.rotational_energy (and mass/CM/momentum).
        Returns: (translational_energy, rotational_energy) as floats (Joules)
        """
        system = self._resolve_input(particles)

        if len(system) == 0:
            self.translational_energy = 0.0
            self.rotational_energy = 0.0
            # ensure masses/cm/momentum cleared
            self._update_mass_and_cm(system)
            return (self.translational_energy, self.rotational_energy)

        # update mass, center of mass and linear momentum attributes
        self._update_mass_and_cm(system)

        # Translational energy of center-of-mass motion
        self.translational_energy = 0.5 * self.total_mass * float(np.dot(self.cm_velocity, self.cm_velocity))

        # Rotational/deformation energy: use velocities relative to CM
        self.rotational_energy = 0.5 * sum(
            p.mass * float(np.dot(p.velocity - self.cm_velocity, p.velocity - self.cm_velocity))
            for p in system
        )

        return (self.translational_energy, self.rotational_energy)

    def linear_momentum(self, particles: Iterable[Particle] = None) -> np.ndarray:
        """
        Compute total linear momentum of the system (kg·m·s^-1).
        If 'particles' is None, use particles stored in the System.
        Updates self.linear_momentum_vec (and mass/CM attributes).
        Returns: numpy array vector of momentum.
        """
        system = self._resolve_input(particles)

        if len(system) == 0:
            self._update_mass_and_cm(system)
            return np.zeros(0, dtype=float)

        # update mass, center of mass and linear momentum attributes
        self._update_mass_and_cm(system)

        return self.linear_momentum_vec

    def angular_momentum(self, particles: Iterable[Particle] = None, r: np.ndarray = np.zeros(0, dtype=float)) -> np.ndarray:
        """
        Compute total angular momentum of the system (Nms) at the specified point r or at the CoM if r is not specified.
        If 'particles' is None, use particles stored in the System.
        Updates self.angular_momentum_vec (and mass/CM attributes).
        Returns: numpy array vector of momentum.
        """
        system = self._resolve_input(particles)

        if len(system) == 0:
            self._update_mass_and_cm(system)
            return np.zeros(0, dtype=float)



        # update mass, center of mass and linear momentum attributes
        self._update_mass_and_cm(system)

        return self.angular_momentum_vec
