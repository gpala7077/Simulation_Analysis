# Name: Gerardo Palacios
# Date: 11/26/2020
# Honor Statement: I have not given or received any unauthorized assistance on this assignment
# YouTube Link: https://youtu.be/DXaabepMEvw


import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class SixSidedDie:
    """A simulation of a six-sided dice.

    Attributes
    ----------
    face_value : int
        Current face value of the simulated dice

    num_sides : int
        Number of sides of the simulated dice"""

    face_value = int()
    num_sides = 6

    def roll(self):
        """Simulate rolling the dice and return face value."""

        self.face_value = random.randint(1, self.num_sides)
        return self.face_value

    def getFaceValue(self):
        """Read the current face value of the dice."""

        return self.face_value

    def __repr__(self):
        """Canonical string representation of class name and current face value"""

        repr_msg = '{}({})'.format(self.__class__.__name__, self.face_value)
        return repr_msg


class TenSidedDie(SixSidedDie):
    """A simulation of a ten-sided dice.

    Attributes
    ----------
    face_value : int
        Current face value of the simulated dice

    num_sides : int
        Number of sides of the simulated dice"""

    num_sides = 10


class TwentySidedDie(SixSidedDie):
    """A simulation of a twenty-sided dice.

    Attributes
    ----------
    face_value : int
        Current face value of the simulated dice

    num_sides : int
        Number of sides of the simulated dice"""

    num_sides = 20


class SimulatedData:
    """Simulation data handler."""

    def __init__(self):
        # initialize empty attributes

        self.parameters = {}
        self.data = pd.DataFrame({
            'actors': [],
            'jailed': [],
            'wealth': [],
            'thieves': [],
            'lieutenants': [],
            'jailed_thieves': [],
            'jailed_lieutenants': [],
            'collected': [],
            'recovered': [],
            'bribes': []
        })
        self.heists = pd.DataFrame({
            'heist_value': [],
            'solve_probability': [],
            'detective_assigned': [],
            'recovered_wealth': [],
        })
        self.weekly_collection = 0
        self.weekly_thieves = 0
        self.weekly_lieutenants = 0
        self.weekly_jailed_thieves = 0
        self.weekly_jailed_lieutenants = 0
        self.weekly_recovered_assets = 0

    def load_parameters(self, filename):
        """Load parameters from work directory csv file."""
        self.parameters = pd.read_csv(filename).to_dict('records')[0]

    def add_new_row(self, df, new_row):
        """Adds new row data frame.

        Parameters
        ----------
        df : str
            Name of data frame to add new row
        new_row
            Data for new row
        """

        if df == 'data':
            self.data = self.data.append(new_row, ignore_index=True)

        elif df == 'heists':
            self.heists = self.heists.append(new_row, ignore_index=True)

    def replace_row(self, df, row_index, new_row, columns=None):
        """Replaces elements for entire row or row and column.

        Parameters
        ----------
        df : str
            Name of data frame

        row_index : int
            Row index for data frame to look at.

        new_row : int or list of ints
            Data to replace row

        columns : str optional(default=None)
            Name of column to replace in row

        """

        if df == 'data':
            if columns is None:
                self.data.loc[row_index] = new_row
            else:
                self.data.loc[row_index:row_index, columns] = new_row

        elif df == 'heists':
            if columns is None:
                self.heists.loc[row_index] = new_row
            else:
                self.heists.loc[row_index:row_index, columns] = new_row

    def get_row(self, df, row_index):
        """Returns a pandas data frame for the given row index.

        Parameters
        ----------
        df : str
            Name of data frame to add new row

        row_index : int
            Row index for data frame to look at.
        """

        if df == 'data':
            return self.data.loc[row_index]

        elif df == 'heists':
            return self.heists.loc[row_index]

    def create_potential_heists(self):
        """Creates a randomly generated data frame for potential heists."""

        # calculate number of potential heists
        num_heists = int(self.data['thieves'].sum() - self.data['jailed_thieves'].sum())

        # Create random data frame
        self.heists = pd.DataFrame({
            'heist_value': [self.parameters['heist_coefficient']*(twenty_sided.roll()**2) for i in range(num_heists)],
            'solve_probability': np.random.rand(num_heists).tolist(),  # Create a list of n random numbers
            'detective_assigned': [-1] * num_heists,  # start out with no detectives assigned to heist
            'recovered_wealth': [0] * num_heists,  # start out with no recovered wealth
        })

    def reset_stats(self):
        """Resets weekly collected stats."""

        self.weekly_collection = 0
        self.weekly_thieves = 0
        self.weekly_lieutenants = 0
        self.weekly_jailed_thieves = 0
        self.weekly_jailed_lieutenants = 0
        self.weekly_recovered_assets = 0

    def save_results(self, filename, columns=('actors', 'jailed', 'wealth', 'bribes')):
        """Save results to csv file

        Parameters
        ----------

        filename : str
            File name including file extension '.csv'
        columns : list of str optional(default=['actors', 'jailed', 'wealth', 'bribes'])
            Columns to be saved to file. Default are actors, jailed, wealth, and bribes

        """
        df = self.data.loc[:, columns]
        df.to_csv(filename)

    def plot_time_series(self, title, stat):
        """Pleasantly displays current board"""

        plt.title(title)
        if isinstance(stat, list):
            for var in stat:
                plt.plot(self.data.index, self.data[var], linestyle='solid')
        else:
            plt.plot(self.data.index, self.data[stat], linestyle='solid')

        plt.yticks(rotation=45)
        plt.xticks(rotation=45)
        plt.show()


class Thief:
    """Low ranking actor in Mr. Biggs crime syndicate

    Attributes
    ----------

    wallet : int
        Collected profits for thief.

    jail : bool
        Boolean if thief has been apprehended
    """

    def __init__(self):
        self.wallet = 0
        self.jail = False
        self.new_recruit()

    def new_recruit(self):
        """Add a new recruit to the weekly stats."""

        simulated_data.weekly_thieves += 1

    def collect_heist(self):
        """Collect bounty from random heist."""

        select = None
        # initialize the total number of heists available
        number_of_heists = int(simulated_data.data['thieves'].sum() - simulated_data.data['jailed_thieves'].sum())

        # Assign a random heist that has not been assigned yet
        while select in crymland.syndicate.heists_assigned or select is None:
            select = random.randint(0, number_of_heists - 1)

        # Record assigned heist
        crymland.syndicate.heists_assigned.update({select: True})
        # Get heist data
        heist_data = simulated_data.get_row('heists', select)
        kickback = self.check_collection(heist_data, select)
        return kickback

    def check_collection(self, heist_data, row_index):
        """Evaluates if thief gets away or gets arrested. Returns kickback to boss.

        Parameters
        ----------
        heist_data : pandas data frame
            Assigned heist data

        row_index : int
            Assigned heist data frame row index
        """

        if heist_data['heist_value'] == -1:  # Detective solved this crime
            taken_away = heist_data['recovered_wealth'] + self.wallet  # Bounty AND wealth is recovered
            simulated_data.replace_row('heists', row_index, taken_away, 'recovered_wealth')  # Saved recovered to df
            simulated_data.weekly_jailed_thieves += 1  # add 1 to weekly stats
            simulated_data.weekly_recovered_assets += self.wallet
            self.wallet = 0  # Zero out wallet
            self.jail = True  # Set jail to true
            return -1

        else:  # Thief got away!
            simulated_data.weekly_collection += heist_data['heist_value']  # Save total collected
            profit = heist_data['heist_value'] * 0.50  # Keep half, kickback half
            self.wallet += profit
            return profit


class Lieutenant(Thief):
    """Mid ranking actor in Mr. Biggs crime syndicate.

    Attributes
    ----------

    wallet : int
        Collected profits for thief.

    jail : bool
        Boolean if thief has been apprehended

    thieves : dict of Thief class
        Dictionary of Thieves being managed.

    lieutenants : dict of Lieutenant class
        Dictionary of Lieutenants being managed.

    in_jail : int
        Number of actors on team that are in jail.

    """

    def __init__(self):
        super().__init__()
        self.thieves = {i: Thief() for i in range(simulated_data.parameters['num_thieves'])}
        self.lieutenants = {}
        self.in_jail = 0
        self.new_recruit()

    def new_recruit(self):
        """Add a new recruit to the weekly lieutenant stats."""

        simulated_data.weekly_lieutenants += 1

    def collect_heist(self):
        """Collect kickbacks from team."""

        collection = self.check_collection(self.check_thieves() + self.check_lieutenants())
        self.promote_thieves()

        return collection

    def check_thieves(self):
        """Collect kickbacks from low ranking actors. Returns total thief kickbacks"""

        collection = 0  # initialize empty collection variable
        remove_thieves = []  # initialize empty list
        for thief in self.thieves:  # iterate through thief list
            value = self.thieves[thief].collect_heist()

            if value > 0:  # Thief gave you proper kickback
                collection += value * 0.50
            else:  # He got caught
                self.in_jail += 1  # Record jailed actor
                remove_thieves.append(thief)  # Add actor in jail to list

        for thief in remove_thieves:  # Remove jailed actors from team
            self.thieves.pop(thief)

        return collection

    def check_lieutenants(self):
        """Collect kickbacks from mid ranking actors. Returns total lieutenant kickbacks"""

        collection = 0  # initialize empty collection variable
        remove_lieutenants = []  # initialize empty list
        for lieutenant in self.lieutenants:  # iterate through lieutenant list
            value = self.lieutenants[lieutenant].collect_heist()  # Check kickbacks

            if value > 0:  # Lieutenant gives proper kickback
                collection += value * 0.50
            else:  # Lieutenant in jail
                self.in_jail += 1
                remove_lieutenants.append(lieutenant)

        for lieutenant in remove_lieutenants:  # Remove jailed lieutenants from team
            self.lieutenants.pop(lieutenant)

        return collection

    def promote_thieves(self):
        """Promote a low-ranking thief to a mid-ranking lieutenant."""
        remove = []  # initialize empty list
        if not self.in_jail:  # Can't promote from jail
            for thief in self.thieves:  # iterate through thieves

                # if thieves have achieved a personal wealth greater than set parameter... promote
                if self.thieves[thief].wallet >= simulated_data.parameters['promotion_wealth']:
                    self.lieutenants.update({thief: Lieutenant()})  # create a new lieutenant
                    self.lieutenants[thief].wallet = self.thieves[thief].wallet  # transfer wealth
                    remove.append(thief)

            for remove_thief in remove:  # remove promoted thief from thief team. No longer commit heists.
                self.thieves.pop(remove_thief)

    def check_collection(self, collection, **kwargs):
        """Checks if collected actors talked in jail. Returns False or kickbacks."""

        # if thieves/LTs have grouped more than given parameter. go to jail.

        if self.in_jail >= simulated_data.parameters['jailed_thieves']:  # Too many members talked
            simulated_data.weekly_jailed_lieutenants += 1
            simulated_data.weekly_recovered_assets += self.wallet + collection
            self.jail = True
            self.wallet = 0
            return -1
        else:
            self.wallet += collection  # Add to personal wealth
            return collection


class Syndicate:
    """Mr. Biggs crime syndicate.

    Attributes
    ----------

    bigg : object
        Object of class Lieutenant. Mr. Bigg.

    heists_assigned : dict
        dictionary of already assigned heists.

    successful_bribes : int
        Number of successful bribes

    """

    def __init__(self):
        self.bigg = Lieutenant()
        self.heists_assigned = {}
        self.successful_bribes = 0

    def reset_stats(self):
        """Reset heists assigned."""

        self.heists_assigned = {}

    def initiate_heists(self):
        """Inform syndicate to being heists."""

        self.bigg.collect_heist()

    def bribes(self):
        """Attempt to bribe detectives that have seized large portion of assets."""

        for detective in crymland.academy.detectives:  # Iterate through detectives
            # Get parameters
            detective = crymland.academy.detectives[detective]
            assets = detective.get_recovered_assets()
            first, after = simulated_data.parameters['seizes_first'], simulated_data.parameters['seizes_thereafter']
            bribe_rate, weekly_collection = simulated_data.parameters['initial_bribe'], simulated_data.weekly_collection

            # Whenever a detective first seizes a total of $1,000,000 from thieves or lieutenants, and every
            # $1,000,000 thereafter, Mr. Bigg attempts to bribe the detective

            if assets >= first and assets >= first + after * self.successful_bribes:
                if detective.evaluate_bribe(bribe_rate * weekly_collection):
                    self.successful_bribes += 1


class Academy:
    """Detective Agency.

    Attributes
    ----------

    detectives : dict
        Dictionary of dedicated detectives.

    """

    def __init__(self):
        self.detectives = {i: Detective() for i in range(simulated_data.parameters['num_detectives'])}
        self.detectives_assigned = {}

    def reset_stats(self):
        """Resets stats."""

        self.detectives_assigned = {}
        for detective in self.detectives:
            self.detectives[detective].bribe_assets = 0

    def assign_heists(self):
        """Assigned detectives on team a heist to investigate."""

        for detective in self.detectives:  # Iterate through each detective
            select = None  # Initialize empty variable
            number_of_heists = int(  # Calculate total number of current heists.
                simulated_data.data['thieves'].sum() - simulated_data.data['jailed_thieves'].sum())

            if detective < number_of_heists:  # As long as there heists left to assign
                while select in self.detectives_assigned or select is None:  # assign a random heist
                    select = random.randint(0, number_of_heists - 1)

                # Record assigned heist
                self.detectives_assigned.update({select: detective})
                # Assign heist
                simulated_data.replace_row('heists', select, detective, 'detective_assigned')

                # Order detective to investigate heist
                self.detectives[detective].investigate(select)

    def review_recovered_assets(self):
        """Review and tally recovered assets from detectives."""

        for detective in self.detectives:  # iterate through detectives
            recovered = simulated_data.heists.loc[  # get data assigned from heist
                simulated_data.heists['detective_assigned'] == detective]

            # Tally the recovered assets
            if recovered['recovered_wealth'].to_list():
                self.detectives[detective].new_recovered_assets(int(recovered['recovered_wealth']))

    def fight_corruption(self):
        """Replace corrupt detectives with an uncorrupted detective."""

        for detective in self.detectives:  # iterate through detectives
            if self.detectives[detective].bribed:  # if corrupted
                search_chance = random.random()

                if search_chance <= self.detectives[detective].discovery_risk:  # attempt to catch
                    self.detectives.update({detective: Detective()})
                else:
                    self.detectives[detective].discovery_risk += twenty_sided.roll() / 100
                    # increase chances of discovery


class Detective:
    """Dedicated detective.

    Attributes
    ----------

    solve_rate : float
        probability detective solves heist.

    solve_cap : float
        max probability detective can solve a heist.

    bribed : bool
        Boolean if detective is bribed.

    bribe_assets : float
        Value of accepted bribe.

    recovered_assets : float
        Value of total recovered assets from investigations.

    discovery_risk : float
        Probability of being caught as a corrupted detective.

    """

    def __init__(self):
        self.solve_rate = simulated_data.parameters['solve_init']
        self.solve_cap = simulated_data.parameters['solve_cap']
        self.bribed = False
        self.bribe_assets = 0
        self.recovered_assets = 0
        self.discovery_risk = 0

    def get_recovered_assets(self):
        """Return the total recovered assets from detective"""

        return self.recovered_assets

    def gain_experience(self):
        """Improve investigation experience."""

        self.solve_rate += ten_sided.roll() / 100  # increase experience

        if self.solve_rate >= self.solve_cap:  # Cap experience
            self.solve_rate = self.solve_cap

    def investigate(self, heist_index):
        """Investigate assigned heist"""

        heist = simulated_data.get_row('heists', heist_index)  # Get heist data
        if not self.bribed and heist['solve_probability'] <= self.solve_rate:  # If not corrupted. Attempt to solve
            saved = heist['heist_value']  # Collect heist_value
            simulated_data.replace_row('heists', heist_index, saved, ['recovered_wealth'])  # Add to recovered wealth
            simulated_data.replace_row('heists', heist_index, -1, ['heist_value'])  # negate heist value
            self.gain_experience()  # Gain experience due to successful investigation

    def new_recovered_assets(self, new):
        """Add to current recovered assets."""

        self.recovered_assets += new
        return self.recovered_assets

    def evaluate_bribe(self, bribe):
        """Evaluate bribe from Mr.Biggs. Returns True if bribe is accepted """

        # Evaluate bribe
        if bribe <= 10000:
            probability = .05

        elif bribe <= 100000:
            probability = .1

        elif bribe <= 1000000:
            probability = .25

        else:
            probability = .5
        # calculate probability
        random_prob = random.random()
        if random_prob <= probability:

            # Accept bribe and become a corrupted detective
            self.bribed = True
            self.bribe_assets += bribe
            self.discovery_risk = simulated_data.parameters['initial_risk']

            return True
        else:
            return False


class Crymland:
    """Land of Crymland.

    Attributes
    ----------

    syndicate : object
        Object of class Syndicate.
    academy : object
        Object of class Academy

    """

    def __init__(self):
        self.syndicate = Syndicate()
        self.academy = Academy()

    def run_sim(self, weeks):
        """Run simulation for the given number of weeks."""

        # Begin with starting stats
        self.record_weekly_stats()
        i = 0
        # Do until end of simulation or until the syndicate has been taken down.
        while i <= weeks and not self.syndicate.bigg.jail:
            simulated_data.create_potential_heists()                        # Create random potential heists
            self.academy.assign_heists()                                    # The Academy assign detectives to the cases
            self.syndicate.initiate_heists()                                # Syndicate begins heists
            self.academy.review_recovered_assets()                          # Detectives review their recovered assets
            self.syndicate.bribes()                                         # Syndicate attempts to bribe detectives
            self.academy.fight_corruption()                                 # The Academy attempts to fight corruption
            self.record_weekly_stats()                                      # Log stats
            i += 1                                                          # Iterate the next week

    def record_weekly_stats(self):
        """Record stats for current week."""

        # Calculate total jailed
        total_jailed = simulated_data.data['jailed_thieves'].sum() + simulated_data.data['jailed_lieutenants'].sum()
        total_jailed += simulated_data.weekly_jailed_thieves + simulated_data.weekly_jailed_lieutenants

        total_actors = simulated_data.data['thieves'].sum() + simulated_data.data['lieutenants'].sum()
        total_actors += simulated_data.weekly_thieves + simulated_data.weekly_lieutenants - 1

        # Create new row to add to data frame
        new_row = {
            'actors': total_actors - total_jailed,
            'jailed': total_jailed,
            'wealth': self.syndicate.bigg.wallet,
            'thieves': simulated_data.weekly_thieves,
            'lieutenants': simulated_data.weekly_lieutenants,
            'jailed_thieves': simulated_data.weekly_jailed_thieves,
            'jailed_lieutenants': simulated_data.weekly_jailed_lieutenants,
            'collected': simulated_data.weekly_collection,
            'recovered': simulated_data.weekly_recovered_assets,
            'bribes': sum([self.academy.detectives[detective].bribe_assets for detective in self.academy.detectives])
        }

        # add new row to data frame
        simulated_data.add_new_row('data', new_row)

        # Reset weekly stats
        simulated_data.reset_stats()
        self.academy.reset_stats()
        self.syndicate.reset_stats()


if __name__ == '__main__':
    # Sample parameters1.csv file to upload data from.

    # weeks,num_thieves,heist_coefficient,promotion_wealth,jailed_thieves,num_detectives,solve_init,solve_cap,seizes_first,seizes_thereafter,initial_risk,initial_bribe
    # 500,7,1000,1000000,3,3,.25,0.75,1000000,1000000,0.05,0.1

    # Generate a twenty sided and ten sided die
    twenty_sided = TwentySidedDie()
    ten_sided = TenSidedDie()

    scenarios = ['parameters/parameters1.csv', 'parameters/parameters2.csv', 'parameters/parameters3.csv']

    for i in range(len(scenarios)):
        # Create Data handler
        simulated_data = SimulatedData()

        # load data parameters
        simulated_data.load_parameters(scenarios[i])

        # Create Crymland
        crymland = Crymland()

        # Run simulation
        crymland.run_sim(simulated_data.parameters['weeks'])

        # Plot results
        simulated_data.plot_time_series('Syndicate Size vs Total Apprehended', ['actors', 'jailed'])
        simulated_data.plot_time_series('Personal Wealth of Mr.Bigg', 'wealth')
        simulated_data.plot_time_series('Accepted Bribes', 'bribes')

        # Save results
        simulated_data.save_results('results/simulation_results{}.csv'.format(i))

