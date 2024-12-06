import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from objective_functionnalities.mkp_functionnalities import select_problem
from Algorithme.article_swarm_optimization_binary import particle_swarm_optimization_binary
from Algorithme.binary_dpso_with_memory import binary_dpso_with_memory
from Algorithme.genetic_algorithm import genetic_algorithm
from Algorithme.genetic_algorithm_with_local_search import genetic_algorithm_with_local_search
from Algorithme.particle_swarm_optimization_binary_condition import particle_swarm_optimization_binary_condition
import pandas as pd  # Ajout de pandas pour manipuler les données

N=30
# Streamlit App
st.set_page_config(
    page_title="Heuristic Optimization Algorithms",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.title("Heuristic Optimization Algorithms 🔍")
st.markdown(
    """
    **Explore and compare various heuristic optimization algorithms for solving multidimensional problems.**
    """
)

# Sidebar for Parameters
st.sidebar.header("Algorithm Parameters ⚙️")
Tmax = st.sidebar.number_input("Maximum Iterations (Tmax)", min_value=1, value=1000, help="The maximum number of iterations.")
step = st.sidebar.number_input("Step Size", min_value=1, value=25, help="The step size between iterations.")
test_runs = st.sidebar.number_input("Number of Test Runs", min_value=1, value=30, help="The number of times each algorithm will run.")
st.sidebar.markdown("---")

# Main UI
st.header("Problem Selection 🎯")
func, D = select_problem()
if func is None or D is None:
    st.error("❌ No problem selected. Please select a valid problem from the functionalities.")
    st.stop()
else:
    st.success(f"✅ Problem selected successfully with dimension: **{D}**")

# Run Algorithms Button
if st.button("Run Optimization 🚀"):
    st.write("### Running Algorithms...")
    progress = st.progress(0)

    results_summary = {}

    def update_progress(percentage):
        progress.progress(percentage)

    try:
        # PSO
        st.write("#### Particle Swarm Optimization (PSO)")
        pso_results = [particle_swarm_optimization_binary(func, D, Tmax, step) for _ in range(test_runs)]
        pso_best = [max(run) for run in pso_results]
        results_summary["PSO"] = {"Best": max(pso_best), "Mean": np.mean(pso_best), "StdDev": np.std(pso_best)}
        st.write(f"✅ Best Solution: {max(pso_best):.5f}, Mean: {np.mean(pso_best):.5f}, StdDev: {np.std(pso_best):.5f}")
        update_progress(20)

        # PSO Condition
        st.write("#### Particle Swarm Optimization with Conditions (PSO_C)")
        pso_c_results = [particle_swarm_optimization_binary_condition(func, D, Tmax, step) for _ in range(test_runs)]
        pso_c_best = [max(run) for run in pso_c_results]
        results_summary["PSO_C"] = {"Best": max(pso_c_best), "Mean": np.mean(pso_c_best), "StdDev": np.std(pso_c_best)}
        st.write(f"✅ Best Solution: {max(pso_c_best):.5f}, Mean: {np.mean(pso_c_best):.5f}, StdDev: {np.std(pso_c_best):.5f}")
        update_progress(40)

        # Genetic Algorithm
        st.write("#### Genetic Algorithm (GA)")
        ga_results = [genetic_algorithm(func, D, Tmax, step) for _ in range(test_runs)]
        ga_best = [max(run) for run in ga_results]
        results_summary["GA"] = {"Best": max(ga_best), "Mean": np.mean(ga_best), "StdDev": np.std(ga_best)}
        st.write(f"✅ Best Solution: {max(ga_best):.5f}, Mean: {np.mean(ga_best):.5f}, StdDev: {np.std(ga_best):.5f}")
        update_progress(60)

        # Genetic Algorithm with Local Search
        st.write("#### Genetic Algorithm with Local Search (GA_LS)")
        gas_results = [genetic_algorithm_with_local_search(func, D, Tmax, step) for _ in range(test_runs)]
        gas_best = [max(run) for run in gas_results]
        results_summary["GA_LS"] = {"Best": max(gas_best), "Mean": np.mean(gas_best), "StdDev": np.std(gas_best)}
        st.write(f"✅ Best Solution: {max(gas_best):.5f}, Mean: {np.mean(gas_best):.5f}, StdDev: {np.std(gas_best):.5f}")
        update_progress(80)

        # BDPSO-M
        st.write("#### Binary DPSO with Memory (BDPSO-M)")
        BDPSO_M_results = [binary_dpso_with_memory(func,N, D, Tmax, step) for _ in range(test_runs)]
        BDPSO_M_best = [max(run) for run in BDPSO_M_results]
        results_summary["BDPSO-M"] = {"Best": max(BDPSO_M_best), "Mean": np.mean(BDPSO_M_best), "StdDev": np.std(BDPSO_M_best)}
        st.write(f"✅ Best Solution: {max(BDPSO_M_best):.5f}, Mean: {np.mean(BDPSO_M_best):.5f}, StdDev: {np.std(BDPSO_M_best):.5f}")
        update_progress(100)

        st.success("🎉 All algorithms executed successfully!")

    # Sauvegarde des résultats
        with open("results.txt", "w") as f:
            f.write("PSO Results:\n")
            for run in pso_results:
                f.write(" ".join(map(str, run)) + "\n")
            f.write("\nPSO_C Results:\n")
            for run in pso_c_results:
                f.write(" ".join(map(str, run)) + "\n")
            f.write("\nGA Results:\n")
            for run in ga_results:
                f.write(" ".join(map(str, run)) + "\n")
            f.write("\nGA_LS Results:\n")
            for run in gas_results:
                f.write(" ".join(map(str, run)) + "\n")
            f.write("\nBDPSO-M Results:\n")
            for run in BDPSO_M_results:
                f.write(" ".join(map(str, run)) + "\n")
        st.write("📄 Results saved to results.txt.")

        
        try:
            with open("results.txt", "r") as f:
                lines = f.readlines()

    # Initialisation des variables pour le traitement
            data = []
            current_algorithm = None

    # Parcourir les lignes du fichier
            for line in lines:
                line = line.strip()
                if line.endswith("Results:"):
                    current_algorithm = line.replace(" Results:", "")
                elif line and current_algorithm:
                    values = list(map(float, line.split()))
                    data.append({"Algorithm": current_algorithm, "Values": values})

    # Convertir les données en DataFrame
            df = pd.DataFrame(data)
            df_values_expanded = df['Values'].apply(pd.Series)
            df_values_expanded.columns = [f"Value_{i+1}" for i in range(df_values_expanded.shape[1])]

            # Combinaison du DataFrame 'Algorithm' avec les colonnes étendues
            df_improved = pd.concat([df['Algorithm'], df_values_expanded], axis=1)

            # Affichage dans Streamlit avec st.dataframe
            st.write("### Résultats améliorés 📊")
            st.dataframe(df_improved)
           

        except Exception as e:
                st.error(f"An error occurred while reading the results file: {str(e)}")

        # Plot Results
        st.write("### Results Comparison 📊")
        plt.figure(figsize=(10, 6))
        plt.plot(range(step, Tmax + 1, step), np.mean(pso_results, axis=0), label='PSO', marker='o')
        plt.plot(range(step, Tmax + 1, step), np.mean(pso_c_results, axis=0), label='PSO_C', marker='v')
        plt.plot(range(step, Tmax + 1, step), np.mean(ga_results, axis=0), label='GA', marker='x')
        plt.plot(range(step, Tmax + 1, step), np.mean(gas_results, axis=0), label="GA_LS", marker='s')
        plt.plot(range(step, Tmax + 1, step), np.mean(BDPSO_M_results, axis=0), label="BDPSO-M", marker='d')
        plt.xlabel("Iterations")
        plt.ylabel("Objective Function Value")
        plt.title("Algorithm Comparisons")
        plt.legend()
        plt.grid()
        st.pyplot(plt)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")