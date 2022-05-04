"""Implementation of Bayesian-optimized charge states.

Automatically assign charges given DFT data. Assume some noise (NOISE)
in the magnetization and  try some number of calls (DEFAULT_CALLS)
for optimization. See jupyter notebook "bayesian-optimization-charge-assignemnts"
for full torial.

References:
    1) Chapter 2.4 of Julia Yang's thesis.
    2) High component paper, submitted.
"""

__author__ = "Julia Yang"


from skopt import gp_minimize
from pymatgen.io.vasp.outputs import *

DEFAULT_CALLS = 50
NOISE = 1E-3
ORDERED = 1.0
NEUTRAL = 0


class DFTProcessor(object):
    """
    Credit: Daniil Kitchaev, circa 2018
    """
    def __init__(self, load_dir, metal, orbital='d'):
        self.load_dir = load_dir
        self.orbital = orbital  # Can specify 's','p', 'd', 'tot'
        self.metal = metal

    def load_dft_runs(self):
        """
        Loads and finds only calculations which have double-relaxed and
        converged.
        """
        data = []
        for root, dirs, files in os.walk(self.load_dir):
            if "OSZICAR" in files and 'CONTCAR' in files and 'OUTCAR' in files and (
                    '3.double_relax' in root or 'relax2' in root):
                try:
                    print("Loading VASP run in {}".format(root))
                    toten = Oszicar(os.path.join(root, 'OSZICAR')).final_energy
                    with open(os.path.join(root, 'OUTCAR')) as outcar:
                        outcar_str = outcar.read()
                    assert ("reached required accuracy" in outcar_str)
                    path = os.sep.join(root.split(os.sep)[0:-1])
                    relaxed = Poscar.from_file(os.path.join(root, "CONTCAR")).structure
                    og_path = os.sep.join(root.split(os.sep)[0:-1])
                    unrelaxed = Poscar.from_file(os.path.join(og_path, "POSCAR")).structure
                    oc = Outcar(os.path.join(root, 'OUTCAR'))
                    magnetizations = []
                    for site_i, site in enumerate(relaxed.sites):
                        if site.species_string in [self.metal]:
                            magnetizations.append(oc.magnetization[site_i][self.orbital])
                    data.append({'s': relaxed.as_dict(),
                                 'toten': toten,
                                 'path': path,
                                 'comp': relaxed.composition.as_dict(),
                                 'unrelaxed': unrelaxed.as_dict(),
                                 'mags': magnetizations})
                except AssertionError:
                    print('Parsing error -- not converged?')
                    continue
        return data


class BayesianChargeAssigner(object):

    def __init__(self, known_species, unknown_species, mag_range, domain):
        """Initialize a BayesianChargeAssigner.

        Args:
           known_species (list of species with charges):
               Provide the species which don't need to oxidation state assignments, e.g.
               [Species('Li', 1), Species('O', -2)]

           unknown_species (list of species with charges that need to be assigned):
               [Species('Mn', 2), Species('Mn', 3), Species('Mn', 4)]

            mag_range (tuple or list of length 2)
                (0,6) units are Bohr magneton

            domain (list of tuples)
                Need to specify loose domain for interpolation. the solution for each cutoff
                will be within the domain specified
                [(2.5, 3.3), (3.7, 4.3), (4.5, 5.0)]
       """
        self.known_species = known_species
        self.num_species_to_assign = len(unknown_species)
        self.mag_range = mag_range
        self.domain = domain
        self.acq_fncs = ['EI', 'LCB', 'PI', 'gp_hedge']
        self.acq_optimizer = 'sampling'

        charges = [i.oxi_state for i in unknown_species]
        self.desired_ox_states = charges

        assert len(set([i.element for i in unknown_species])) == 1
        self.metal = unknown_species[0].element.name
        self.metal_charges = [i.to_pretty_string() for i in unknown_species]

        self.cn = None
        self.optimization_results = None

    def objective_fnc(self, current_params):
        """Returns number of structures which are not charge-balanced."""
        self.cn.set_parameters(current_params)
        return self.cn.evaluate()

    def optimize_charges(self, structure_list):
        """
        Returns:
            dict: acquisition function and its number of non-charge-balanced-structures, upper cutoffs
        """
        optimization_results = dict()
        s_ij, q_ij = [], []
        for struct_data in structure_list:
            if 'mags' not in struct_data:
                # add a message here
                return None
            q_total = self.add_up_known_charges(Structure.from_dict(struct_data['s']))
            s_ij.append(struct_data['mags'])
            q_ij.append(q_total)
        init_cutoffs = [self.mag_range[0] + self.mag_range[1] /
                        float(self.num_species_to_assign)
                        * i for i in range(self.num_species_to_assign)]
        self.cn = ChargeNeutrality(s_ij, q_ij, init_cutoffs,
                                   self.desired_ox_states)
        for fnc in self.acq_fncs:
            residual = gp_minimize(self.objective_fnc,
                                   self.domain,
                                   acq_func=fnc,
                                   n_calls=DEFAULT_CALLS,
                                   acq_optimizer=self.acq_optimizer,
                                   noise=NOISE)
            optimization_results[fnc] = [residual.fun, residual.x]
        return optimization_results

    def assign_charges(self, data):
        """Returns dictionary of charge-assigned, charge-balanced structures."""
        self.optimization_results = self.optimize_charges(data)
        sorted_cutoffs = list(sorted(self.optimization_results.items(),
                                     key=lambda item: item[1][1]))
        optimized_cutoffs = sorted_cutoffs[0][1][1]  # Since our structure is: [acq_function, [minimum, cutoff_list]]
        return self.do_charge_balancing(data, optimized_cutoffs)

    def do_charge_balancing(self, data, cutoffs):
        """Takes Bayesian-optimized cutoffs and assigns charges."""
        known_oxi_states = {i.element.name: {i.to_pretty_string(): ORDERED}
                            for i in self.known_species}
        charge_balanced = []

        for s_data in data:
            s = Structure.from_dict(s_data['s'])
            s.remove_oxidation_states()
            s.replace_species(known_oxi_states)
            metal_sites = self.get_site_index(s, self.metal)
            for i, mag in enumerate(s_data['mags']):
                site = metal_sites[i]
                if mag < np.min(cutoffs):
                    s[site] = self.metal_charges[0]
                elif mag > np.max(cutoffs):
                    s[site] = self.metal_charges[-1]
                else:
                    for i_cutoff in range(len(cutoffs) - 1):
                        if cutoffs[i_cutoff] < mag < cutoffs[i_cutoff + 1]:
                            s[site] = self.metal_charges[i_cutoff + 1]
            if s.charge == NEUTRAL:
                s_data['s'] = s.as_dict()
                charge_balanced.append(s_data)
        return charge_balanced

    @staticmethod
    def get_site_index(s, chem):
        """Simple method to get site indices within a structure."""
        sites = []
        for i, site in enumerate(s):
            if site.species_string == chem:
                sites.append(i)
        return sites

    def add_up_known_charges(self, structure):
        """Calculates structure charge for the known species."""
        total_charge = 0
        default_ox = {i.element.name: i.oxi_state for i in self.known_species}
        for i, site in enumerate(structure):
            if site.species_string in default_ox:
                total_charge += default_ox[site.species_string]
        return total_charge


class ChargeNeutrality(object):

    def __init__(self, x, y, cutoffs, charges):
        """
        Args:
            x: (list of lists)
                All magnetic moments within each structure

            y: list
                Current structure total charge, accounting for only charges from known species.
                Typically a list of all negative values

            cutoffs (list)
                [cutoff1, cutoff2, cutoff3] for the current optimization loop.

            charges (list)
                [4, 3, 2] charge for the metals, in the same order as unknown_species specified for
                 BayesianChargeAssigner
            """
        self.cutoffs = cutoffs
        self.charges = charges
        self.sij = x
        self.q_tot = y

    def map(self, mag):
        """
        Args:
            mag (float)
            Single magnetization for a site within a structure

        Returns:
              charge state (float)
        """
        if mag < np.min(self.cutoffs):
            return self.charges[0]
        elif mag > np.max(self.cutoffs):
            return 10**10
        for i_uB in range(1, len(self.cutoffs)):
            if self.cutoffs[i_uB-1] < mag < self.cutoffs[i_uB]:
                return self.charges[i_uB]

    def set_parameters(self, current_params):
        """Sets current cutoffs."""
        self.cutoffs = current_params

    def evaluate(self):
        """Calculate number of non-charge-balanced structures."""
        total = 0
        for index in range(len(self.sij)):
            struct = self.sij[index]
            s_charge = 0
            for mag in struct:
                s_charge += self.map(np.abs(mag))
            if (s_charge + self.q_tot[index]) != 0:
                total += 1
        return total
