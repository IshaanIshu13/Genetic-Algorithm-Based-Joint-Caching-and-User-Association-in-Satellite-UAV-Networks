import math
import numpy as np
import random

# Constants----------------save
beta = 2  #Path loss exponent
nlos = 1
nnlos = 20
g_su = 0.68783 #2.0635 #0.78 Satellite to UAV fading factor
g_cs = 0.64  #UAV to UAV fading factor
g_u1_u = 0.86  #UAV to UAV fading factor
g_i_u = 0.69  #UAV to user fading factor
f_c = 20 * 10**9  #20 GHz in Hz
N = 800 # no. of users 
U = 80
S = 1
Nu = 16
M = 4
F = 30
#FOR ADDING HOVERING ENERGY
Hv = 500       # UAV height in meters
W =  2.038  # convert to Newtons
rd = 0.26       # Rotor disk radius in meters 

file_size = 100 * 1024  # Assume 100 Kb file size
tau = 0.0001 # sec
phi = 2  # Placeholder for computation energy factor
C_k = np.random.rand(U) * 1000  # Placeholder for computation capacity of satellites (random values)
c = 3e8  # Speed of light in meters per second (m/s)
pi_value = math.pi
B = 100 * 10**6  # Transmission bandwidth in Hz (100 MHz)
transmit_power_sat_max = 50  # in dBm
transmit_power_uav_max = 37  # in dBm
receive_power_uav = 20 # in dBm

noise_power = -174 + 10 * math.log10(B)
sigma_squared = 10**((noise_power - 30) / 10)  # Convert noise power from dBm to linear scale

s_s = 780000  # Satellite Altitude in meters
s_u = 500     # UAV altitude in meters
uavs_per_region = U // 4  # UAVs per region
z = 10 ** -27

# Convert dBm to Watts for satellite and UAV transmit power
P_sat = 10 ** ((transmit_power_sat_max - 30) / 10)  # Satellite transmit power in Watts
P_uav = 10 ** ((transmit_power_uav_max - 30) / 10)  # UAV transmit power in Watts
P_uavr = 10 ** ((receive_power_uav - 30) / 10)  # UAV transmit power in Watts

# Initialize positions for users, UAVs, and satellite
user_pos = [[np.random.randint(1, 10001), np.random.randint(1, 10001), 0] for _ in range(N)]
sat_pos = [[5000, 5000, s_s]]  # Satellite at center

uav_pos = [[] for _ in range(4)]  # Initialize empty lists for each region
# Generate UAV positions for four regions
# Region 1 (Top-left): x in [0, 5000], y in [5000, 10000]
for _ in range(uavs_per_region):
    uav_pos[0].append([np.random.randint(0, 5001), np.random.randint(5000, 10001), s_u])

# Region 2 (Top-right): x in [5000, 10000], y in [5000, 10000]
for _ in range(uavs_per_region):
    uav_pos[1].append([np.random.randint(5000, 10001), np.random.randint(5000, 10001), s_u])

# Region 3 (Bottom-left): x in [0, 5000], y in [0, 5000]
for _ in range(uavs_per_region):
    uav_pos[2].append([np.random.randint(0, 5001), np.random.randint(0, 5001), s_u])

# Region 4 (Bottom-right): x in [5000, 10000], y in [0, 5000]
for _ in range(uavs_per_region):
    uav_pos[3].append([np.random.randint(5000, 10001), np.random.randint(0, 5001), s_u])
# ---------- Hovering Energy Function ---------------------------------
  
def hovering_power(Hv, W, rd, c1=1.91, c2=1.1, xi0=1.225):
    xi = xi0 * np.exp(-1.18e-4 * Hv)
    return (c1 * xi + c2 * W**1.5) /np.sqrt(xi * np.pi * rd**2)

def hovering_energy(Hv, W, rd, duration):
    #print("hovering_energy() called with duration:", duration)  # Your added print
    return hovering_power(Hv, W, rd) * duration

# ---------- Hovering Energy Function Ends-----------------------------------


# Function to calculate Euclidean distance
def calculate_distance(point1, point2):
    distance = np.sqrt((point1[0] - point2[0]) ** 2 +
                  (point1[1] - point2[1]) ** 2 +
                  (point1[2] - point2[2]) ** 2)
    return distance

def calculate_channel_gain_uav_user(P_t, distance, g_tr, beta):
    #Probability of LoS and NLoS
    P_Los = random.uniform(0, 1)
    #P_Los = (1 + 9.61 * math.exp(-0.16 * math.atan(500 / distance)) - 9.61)-1
    P_NLos = 1 - P_Los

    # Calculate los without exponentiation
    los = g_tr * (4 * pi_value * f_c * distance / c) ** (-beta / 2)

    # Calculate the path loss due to LoS and NLoS
    path_loss_factor = 10 ** (-(nlos * P_Los + nnlos * P_NLos) / 20)

    # Final gain calculation
    gain_uav_user = los * path_loss_factor
    #print(f"UAV-User Channel Gain: Distance = {distance}, Gain = {gain_uav_user}")

    return gain_uav_user

# Function to calculate channel gain between transmitter and receiver
def calculate_channel_gain_sat_uav(P_t, distance, g_tr, beta):
    # Channel gain formula: h = g_tr * d^(-beta)
    return g_tr * np.sqrt(P_sat * distance ** (-beta))
    #return np.sqrt(g_tr * (distance ** (-beta)))

# Function to calculate channel gain between transmitter and receiver
def calculate_channel_gain_uav_uav(P_t, distance, g_tr, beta):
    #Channel gain formula: h = g_tr * d^(-beta)
    return g_tr * np.sqrt(P_uav * distance ** (-beta))

# Function to calculate data rate based on channel gain
def calculate_data_rate(P_t, channel_gain):
    #Data rate formula: R = B * log2(1 + SNR), SNR = P_t * b / sigma_squared
    SNR = (P_t * (channel_gain ** 2)) / sigma_squared
    return B * np.log2(1 + SNR)

# Initialize channel gains and data rates lists
sat_uav_channel_gains = []
sat_uav_data_rates = []
uav_uav_channel_gains = []
uav_uav_data_rates = []
user_uav_channel_gains = []
user_uav_data_rates = []

# 1. Calculate Satellite to UAV data rate and channel gain
for region in uav_pos:
    for uav in region:
        distance = calculate_distance(sat_pos[0], uav)
        gain = calculate_channel_gain_sat_uav(P_sat, distance, g_su, beta)
        rate = calculate_data_rate(P_sat, gain)
        #print("satellite to UAV data rate", rate)
        sat_uav_channel_gains.append(gain)
        sat_uav_data_rates.append(rate)

#data_rate_gbps_sat_uav = sat_uav_data_rates / 1e9
#print("satellite to UAV rate", sat_uav_data_rates)

# 2. Calculate UAV to UAV data rate and channel gain within regions
for region in uav_pos:
    region_gains = []
    region_rates = []
    for i in range(len(region)):
        for j in range(i + 1, len(region)):
            distance = calculate_distance(region[i], region[j])
            gain = calculate_channel_gain_uav_uav(P_uav, distance, g_u1_u, beta)
            rate = calculate_data_rate(P_uav, gain)
            #print("UAV to UAV data rate", rate)
            region_gains.append(gain)
            region_rates.append(rate)
    uav_uav_channel_gains.append(region_gains)
    uav_uav_data_rates.append(region_rates)
    #print(f"  Distance: {distance:.2f} meters, Channel Gain: {gain:.6f}, Data Rate: {rate:.2f} bps")
    #print("UAV to UAV data rate", uav_uav_data_rates)

# 3. UAV to User data rate and channel gain
for i, user in enumerate(user_pos):
    user_region = 0 if user[0] <= 5000 and user[1] >= 5000 else \
                  1 if user[0] > 5000 and user[1] >= 5000 else \
                  2 if user[0] <= 5000 else 3
    user_gains = []
    user_rates = []
    for j, uav in enumerate(uav_pos[user_region]):
        distance = calculate_distance(user, uav)
        gain = calculate_channel_gain_uav_user(P_uav, distance, g_i_u, beta)
        rate = calculate_data_rate(P_uav, gain)
        #print("UAV to user data rate", rate)
        user_gains.append(gain)
        user_rates.append(rate)
    user_uav_channel_gains.append(user_gains)
    user_uav_data_rates.append(user_rates)

# Cache status matrix
Gamma = np.zeros((F, U), dtype=int)  # Cache status matrix
Gamma1 = np.zeros((F, U), dtype=int)  # Cache status matrix

# Function to generate random cache matrix
def generate_cache_matrix(F, U, M):
    matrix = np.zeros((F, U), dtype=int)  # Initialize with zeros
    for uav in range(U):
        indices = np.random.choice(F, size= M, replace=False)  # Randomly choose cache_capacity files
        matrix[indices, uav] = 1
    return matrix

Gamma1 = generate_cache_matrix(F, U, M)
Gamma = Gamma1


# Initialize primary UAVs for users based on region and UAV capacity
roh = np.zeros((N, U))  # UAV to user association matrix
primary_uav_capacity = [Nu] * U  # Track capacity for each UAV

# UAV to user association matrix
roh = np.zeros((N, U))  # N users, U UAVs

def get_user_region(user):
    """
    Determine the region of the user based on their position.
    Regions are divided into quadrants.
    """
    if user[0] <= 5000 and user[1] >= 5000:
        return 0  # Region 1 (Top-left)
    elif user[0] > 5000 and user[1] >= 5000:
        return 1  # Region 2 (Top-right)
    elif user[0] <= 5000 and user[1] < 5000:
        return 2  # Region 3 (Bottom-left)
    else:
        return 3  # Region 4 (Bottom-right)


# Function to initialize primary UAVs for userbs ased on region and UAV capacity
def initialize_primary_uav(roh, user_pos, uav_pos, Nu, U, uavs_per_region):
    primary_uav_capacity = [Nu] * U  # Track capacity for each UAV
    global_user_uav_dict = {}
    for i, user in enumerate(user_pos):
        #print("i =", i, user)
        region = get_user_region(user)  # Get the user's region
        #print("region", region)
        uav_indices = list(range(region * uavs_per_region, (region + 1) * uavs_per_region))  # UAVs in the region
        #print("region * uavs_per_region = ", region * uavs_per_region)
        #print("UAV indices", uav_indices)

        # Sort UAVs based on distance from the user (ascending order)
        sorted_uav_indices = sorted(uav_indices, key=lambda uav_index: calculate_distance(user, uav_pos[region][uav_index % uavs_per_region]))
        #print("sorted UAV indices", sorted_uav_indices)
        temp = tuple(sorted_uav_indices)
        try:
            global_user_uav_dict[temp]
        except KeyError:
            global_user_uav_dict[temp] = sorted_uav_indices

        # Try to assign the nearest available UAV
        assigned = False
        for uav_index in sorted_uav_indices:
            if primary_uav_capacity[uav_index] > 0:
                roh[i][uav_index] = 1  # Set as primary UAV
                primary_uav_capacity[uav_index] -= 1
                assigned = True
                break

        # If no UAV was assigned, attempt to assign the next farthest UAV
        if not assigned:
            if len(global_user_uav_dict[temp]) > 0:
              global_user_uav_dict[temp].pop(0)
              for uav_index in sorted_uav_indices:  # Recheck all UAVs in sorted order
                  if primary_uav_capacity[uav_index] > 0:
                      roh[i][uav_index] = 1  # Set as primary UAV
                      primary_uav_capacity[uav_index] -= 1
                      assigned = True
                      break

        if not assigned:
            print(f"User {i} could not be assigned a UAV within its region due to capacity limits")

    #print("Roh matrix after initialization:")
    #print(roh)  # Debugging: Check the roh matrix after assignment
    return roh


#Assign primary UAVs to users
roh = initialize_primary_uav(roh, user_pos, uav_pos, Nu, U, uavs_per_region)

# Print the roh matrix
print("Roh Matrix (Primary UAV assignments):\n", roh)

# Cluster list to group users by UAV
clusters = [[] for _ in range(U)]
for i, user_assignment in enumerate(roh):
    for j, is_primary in enumerate(user_assignment):
        if is_primary:
            clusters[j].append(i)  # Append user index to the corresponding UAV's cluster

# Print cache matrix and clusters
print("Cache Matrix (Gamma):\n", Gamma)
print("\nClusters of users for each UAV:")
for i, cluster in enumerate(clusters):
    print(f"UAV {i}: Users {cluster}")

# Function to calculate transmission delay
def calculate_transmission_delay(file_size, data_rate):
    # Transmission delay (T) = file_size / data_rate
    return (file_size / data_rate)

# Function to calculate energy consumption
def calculate_energy_consumption(P_t, transmission_delay):
    # Energy consumption (E) = Power * Time
    return P_t * transmission_delay
freq_cpu = 3000000000 # in Hz
cpu_cycles = 100 # cycle per bit

# change for hover

def assign_secondary_and_calculate(N, F, roh, Gamma, clusters, user_uav_data_rates, uav_uav_data_rates, sat_uav_data_rates, file_size, tau, phi):
    total_delay = np.zeros(N)
    total_energy = np.zeros(N)
    for user in range(N):
        # region_of_req_user = get_user_region(user)
        # print("region_of_req_user =", region_of_req_user)
        primary_uav = -1
        # Find primary UAV for the user
        for uav, is_primary in enumerate(roh[user]):
            if is_primary:
                primary_uav = uav
                break

        if primary_uav == -1:
            continue  # No primary UAV found

        # Check if the file is cached in the primary UAV
        requested_file = np.random.randint(0, F)  # Randomly select a file to request
        #print("Requested file =", requested_file)
        #print("user requested for the file =", user)
        #print("primary UAV", primary_uav)
        if Gamma[requested_file, primary_uav] == 1:
            # Case 1: File is cached in the primary UAV
            data_rate_primary_user = user_uav_data_rates[user][primary_uav % uavs_per_region]
            transmission_delay = calculate_transmission_delay(file_size, data_rate_primary_user)
            # change start
            transmit_energy = calculate_energy_consumption(P_uav, transmission_delay)
            hover_energy = hovering_energy(Hv, W, rd, transmission_delay)
            energy_consumption = transmit_energy + hover_energy
            total_delay[user] = transmission_delay
            total_energy[user] = energy_consumption
            #print(f"User {user} Primary UAV Hover Energy: {hover_energy:.6f} J")
            # change over

        else:
            # Case 2: File is not in the primary UAV, check secondary UAVs
            region = primary_uav // uavs_per_region  # Get the region of the primary UAV

            #print("region of primary UAV =", region)

            uav_indices_in_region = list(range(region * uavs_per_region, (region + 1) * uavs_per_region))
            secondary_found = False
            for secondary_uav in uav_indices_in_region:
                if secondary_uav != primary_uav and Gamma[requested_file, secondary_uav] == 1:
                    # File is cached in secondary UAV
                    data_rate_primary_user = user_uav_data_rates[user][primary_uav % uavs_per_region]
                    data_rate_secondary_primary = uav_uav_data_rates[primary_uav // uavs_per_region][secondary_uav % uavs_per_region]

                    delay_secondary_to_primary = calculate_transmission_delay(file_size, data_rate_secondary_primary)
                    delay_primary_to_user = calculate_transmission_delay(file_size, data_rate_primary_user)

                    transmission_delay = (delay_secondary_to_primary + delay_primary_to_user)

                    receiving_energy = calculate_energy_consumption(P_uavr, delay_secondary_to_primary)
                    transmission_energySectoPri = calculate_energy_consumption(P_uav, delay_secondary_to_primary)

                    #change start
                    hover_energy_secondary_to_primary = hovering_energy(Hv, W, rd, delay_secondary_to_primary)

                    energy_consumption = (receiving_energy +
                      transmission_energySectoPri +
                      calculate_energy_consumption(P_uav, delay_primary_to_user) +
                      hover_energy_secondary_to_primary)
                    total_delay[user] = transmission_delay
                    total_energy[user] = energy_consumption

                    #print(f"User {user} Secondary UAV Hover Energy: {hover_energy_secondary_to_primary:.6f} J (Sec->Pri), "
      #f"{hover_energy_primary_to_user:.6f} J (Pri->User)")

                    #change end
                    
                    
                    
                    secondary_found = True
                    break

            if not secondary_found:
                # Case 3: Fetch from satellite
                data_rate_sat_primary = sat_uav_data_rates[primary_uav]
                data_rate_primary_user = user_uav_data_rates[user][primary_uav % uavs_per_region]

                delay_satellite_to_primary = calculate_transmission_delay(file_size, data_rate_sat_primary)
                delay_primary_to_user = calculate_transmission_delay(file_size, data_rate_primary_user)
                computational_delay = (file_size * cpu_cycles) / freq_cpu  # Placeholder for computational delay
                #computational_energy = z * file_size * cpu_cycles * freq_cpu * freq_cpu  # Placeholder for computational energy consumption

                computational_energy = calculate_energy_consumption(P_sat, computational_delay)
                #print("computational energy =", computational_energy)
                #print("computational delay =", computational_delay)

                transmission_delay = (delay_satellite_to_primary + delay_primary_to_user + computational_delay)

                receive_energy = calculate_energy_consumption(P_uavr, delay_satellite_to_primary)
                #change start
                total_hover_time = delay_satellite_to_primary + delay_primary_to_user
                hover_energy = hovering_energy(Hv, W, rd, total_hover_time)

                energy_consumption = (calculate_energy_consumption(P_sat, delay_satellite_to_primary) +
                      receive_energy +
                      calculate_energy_consumption(P_uav, delay_primary_to_user) +
                      computational_energy +
                      hover_energy)
                total_delay[user] = transmission_delay
                total_energy[user] = energy_consumption
                #print(f"User {user} Satellite Delivery Hover Energy: {hover_energy:.6f} J over {total_hover_time:.6f} s")

                #change end
                

                
                #print("Delay for user when served from satellite", total_delay[user])
                

    return total_delay, total_energy
# changeeee over

# Assuming tau, phi are defined in your context
total_delay, total_energy = assign_secondary_and_calculate(
    N, F, roh, Gamma, clusters, user_uav_data_rates, uav_uav_data_rates, sat_uav_data_rates, file_size, tau, phi
)
print("\nTotal Delay in sec for each user:")
print(total_delay)
#print("\nEnergy Consumption (Joules) for each user:")
#print(total_energy)
total_delay_all = np.sum(total_delay)
print(f"\nTotal Delay to Serve All Users: {total_delay_all} sec")

# Calculate service cost for each user
service_cost = total_energy * phi

# Print service cost for each user
print("\nService Cost (Currency) for each user:")
print(service_cost)

# Calculate total service cost for all users
total_service_cost = np.sum(service_cost)
print(f"\nTotal Service Cost to Serve All Users: {total_service_cost} dollar")

## Priority weight factors
## priority_weights = {
##     1: 1.0,  # Highest priority
##     2: 1.2,  # High priority
##     3: 1.5,  # Medium priority
##     4: 2.0   # Low priority
## }
#
## Assign priorities to users
## For simplicity, we'll randomly assign a priority to each user
#user_priorities = np.random.choice([1, 2, 3, 4], size=N)
#
## Calculate service cost with priority factors
#def calculate_service_cost_with_priority(total_energy, phi):
#    #service_cost = np.zeros(N)
#    for user in range(N):
#        # priority = user_priorities[user]  # Get the user's priority
#        # omega = priority_weights[priority]  # Get the weight factor based on priority
#        service_cost[user] = total_energy[user] * phi
#    #print(f"User {user}: Priority {user_priorities[user]}, Service Cost: {service_cost_with_priority[user]}")
#    return service_cost
#
## Calculate the service cost with the new priority factors
#service_cost_with_priority = calculate_service_cost_with_priority(total_energy, phi)
#
## Print the service costs and the corresponding priorities
##print("\nService Cost (with priority) for each user:")
#for user in range(N):
#    print(f"User {user}: Service Cost: {service_cost_with_priority[user]}")
#
## Calculate total service cost for all users
#total_service_cost_with_priority = np.sum(service_cost_with_priority)
#print(f"\nTotal Service Cost to Serve All Users (with priority): {total_service_cost_with_priority} dollar")

def enforce_roh_constraints_after_toggle(roh, Nu, U, user_pos, uav_pos, uavs_per_region, user_uav_data_rates):
    # Initialize an array to count the number of primary users assigned to each UAV
    uav_user_count = np.zeros(U, dtype=int)

    # Step 1: Assign the UAV with the maximum data rate as primary for each user within their region
    for user in range(roh.shape[0]):
        # Find UAVs assigned as primary for this user
        primary_uavs = np.where(roh[user] == 1)[0]

        # Filter the UAVs to only those in the user's region
        region = get_user_region(user_pos[user])
        region_uavs = [uav for uav in primary_uavs if region * uavs_per_region <= uav < (region + 1) * uavs_per_region]

        if len(region_uavs) > 1:
            # Retrieve data rates for each UAV in the user's region
            data_rates = [user_uav_data_rates[user][uav % uavs_per_region] for uav in region_uavs]

            if data_rates:
                # Select the UAV with the highest data rate for this user
                best_uav = region_uavs[np.argmax(data_rates)]

                # Set only the best UAV as primary in roh, others are set to 0
                roh[user] = 0
                roh[user][best_uav] = 1

                # Increment the user count for the selected UAV
                uav_user_count[best_uav] += 1
            else:
                print(f"Warning: No valid data rates found for user {user} in region {region}.")

    # Step 2: Enforce the max Nu users per UAV constraint
    for uav in range(U):
        if uav_user_count[uav] > Nu:
            # Find all users assigned to this UAV
            assigned_users = np.where(roh[:, uav] == 1)[0]

            # Retrieve data rates for each assigned user with respect to this UAV
            data_rates = [user_uav_data_rates[user][uav % uavs_per_region] for user in assigned_users]

            # Sort users by data rate in descending order (higher data rate users have priority)
            sorted_users = sorted(zip(data_rates, assigned_users), reverse=True)

            # Keep only the top Nu users for this UAV, deassign others
            for _, user in sorted_users[Nu:]:
                roh[user][uav] = 0  # Remove this user from the UAV
                uav_user_count[uav] -= 1  # Adjust the UAV's user count

    return roh




# ========== GENETIC ALGORITHM FOR CACHING OPTIMIZATION ==========             
# GA Parameters
population_size = 10
generations = 20
mutation_rate = 0.1            
def initialize_population(pop_size, F, U, M):
    return [generate_cache_matrix(F, U, M) for _ in range(pop_size)]               
def fitness_function(Gamma_candidate):
    # Reuse the existing clustering and roh structure
    delay, energy = assign_secondary_and_calculate(
        N, F, roh, Gamma_candidate, clusters, user_uav_data_rates,
        uav_uav_data_rates, sat_uav_data_rates, file_size, tau, phi
    )
    cost = np.sum(energy * phi)
    return -cost  # Negative because we minimize cost              
def crossover(parent1, parent2):
    child = np.copy(parent1)
    for f in range(F):
        for u in range(U):
            if random.random() > 0.5:
                child[f][u] = parent2[f][u]
    return child               
def mutate(candidate, mutation_rate, M):
    for u in range(U):
        if random.random() < mutation_rate:
            # Reset UAV cache and reselect M files
            candidate[:, u] = 0
            files = np.random.choice(F, M, replace=False)
            candidate[files, u] = 1
    return candidate               
# GA main loop
population = initialize_population(population_size, F, U, M)
best_fitness = float('-inf')
best_solution = None               
for gen in range(generations):
    # Evaluate fitness
    fitness_scores = [fitness_function(ind) for ind in population]
    ranked_population = [x for _, x in sorted(zip(fitness_scores, population), key=lambda pair: pair[0], reverse=True)]            
    # Elitism: retain top 2
    new_population = ranked_population[:2]             
    # Generate rest via crossover and mutation
    while len(new_population) < population_size:
        parents = random.sample(ranked_population[:5], 2)
        child = crossover(parents[0], parents[1])
        child = mutate(child, mutation_rate, M)
        new_population.append(child)               
    population = new_population            
    # Track best solution
    if max(fitness_scores) > best_fitness:
        best_fitness = max(fitness_scores)
        best_solution = ranked_population[0]               
    print(f"Generation {gen+1}: Best Fitness = {-best_fitness:.2f} (lower is better)")             
# Apply the best caching decision
Gamma = best_solution
print("\nBest Gamma (cache matrix) found by GA:")
print(Gamma)               
# Recalculate final cost/delay
final_delay, final_energy = assign_secondary_and_calculate(
    N, F, roh, Gamma, clusters, user_uav_data_rates,
    uav_uav_data_rates, sat_uav_data_rates, file_size, tau, phi
)
final_cost = np.sum(final_energy * phi)
print(f"\nFinal Total Service Cost After GA Optimization: {final_cost:.2f} dollar")            
