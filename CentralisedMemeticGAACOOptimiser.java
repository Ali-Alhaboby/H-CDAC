package hu.u_szeged.inf.fog.simulator.workflow.aco;

import hu.mta.sztaki.lpds.cloud.simulator.util.SeedSyncer;
import hu.u_szeged.inf.fog.simulator.node.WorkflowComputingAppliance;

import java.util.*;

public class CentralisedMemeticGAACOOptimiser {

    public static HashMap<Integer, ArrayList<WorkflowComputingAppliance>> runOptimiser(
            int clusterCount,
            ArrayList<WorkflowComputingAppliance> nodes,

            // GA
            int populationSize,
            int generations,
            double crossoverRate,
            double mutationRate,
            int tournamentSize,
            int elitismCount,

            // Memetic control
            int refineEveryR,
            int refineEliteCount,

            // ACO local refinement
            int acoAnts,
            int acoIters,
            double acoProbability,
            double acoTopPercent,
            double acoPheromoneIncrement,
            double acoEvaporationRate,

            // Seeding strength
            double seedBoost // مثال: 0.35
    ) {

        int n = nodes.size();
        Individual[] pop = new Individual[populationSize];

        for (int i = 0; i < populationSize; i++) {
            pop[i] = Individual.random(n, clusterCount);
            pop[i].fitness = fitness(clusterCount, nodes, pop[i].genes);
        }

        Individual globalBest = bestOf(pop).copy();

        for (int gen = 1; gen <= generations; gen++) {

            // ---- GA generation ----
            Arrays.sort(pop, Comparator.comparingDouble(o -> o.fitness));
            Individual[] next = new Individual[populationSize];

            // elitism
            for (int e = 0; e < elitismCount; e++) next[e] = pop[e].copy();

            // rest
            for (int i = elitismCount; i < populationSize; i++) {
                Individual p1 = tournament(pop, tournamentSize);
                Individual p2 = tournament(pop, tournamentSize);

                Individual child = (SeedSyncer.centralRnd.nextDouble() < crossoverRate)
                        ? uniformCrossover(p1, p2)
                        : p1.copy();

                mutate(child, clusterCount, mutationRate);
                child.fitness = fitness(clusterCount, nodes, child.genes);
                next[i] = child;
            }

            pop = next;

            Individual genBest = bestOf(pop);
            if (genBest.fitness < globalBest.fitness) globalBest = genBest.copy();

            // ---- Memetic ACO refinement every R generations ----
            if (refineEveryR > 0 && gen % refineEveryR == 0) {
                Arrays.sort(pop, Comparator.comparingDouble(o -> o.fitness));
                int E = Math.min(refineEliteCount, pop.length);

                // map node->index for decoding refined clusters back to genes
                Map<WorkflowComputingAppliance, Integer> index = new HashMap<>();
                for (int i = 0; i < nodes.size(); i++) index.put(nodes.get(i), i);

                for (int ei = 0; ei < E; ei++) {
                    Individual elite = pop[ei];

                    double[][] seeded = buildSeededPheromone(n, clusterCount, elite.genes, seedBoost);

                    HashMap<Integer, ArrayList<WorkflowComputingAppliance>> refinedClusters =
                            CentralisedAntOptimiser5.runOptimiserSeeded(
                                    clusterCount, nodes,
                                    acoAnts, acoIters,
                                    acoProbability, acoTopPercent,
                                    acoPheromoneIncrement, acoEvaporationRate,
                                    seeded
                            );

                    int[] refinedGenes = decodeGenesFromClusters(refinedClusters, index, n);

                    double refinedFitness = fitness(clusterCount, nodes, refinedGenes);

                    if (refinedFitness < elite.fitness) {
                        elite.genes = refinedGenes;
                        elite.fitness = refinedFitness;

                        if (elite.fitness < globalBest.fitness) {
                            globalBest = elite.copy();
                        }
                    }
                }
            }
        }

        return buildClusters(clusterCount, nodes, globalBest.genes);
    }
    
    
    

    // -------- Seeding pheromone around GA solution --------
    // base ~ 0.45..0.55 مثل تهيئة ACO الأصلية 
    private static double[][] buildSeededPheromone(int n, int k, int[] genes, double boost) {
        double[][] m = new double[n][k];
        for (int i = 0; i < n; i++) {
            // base noise
            for (int c = 0; c < k; c++) {
                double noise = SeedSyncer.centralRnd.nextDouble() * 0.1;
                m[i][c] = 0.5 - 0.05 + noise; // ~ [0.45,0.55]
            }
            // boost the chosen cluster
            int chosen = genes[i];
            m[i][chosen] += boost;
        }
        return m;
    }

    // -------- Fitness (same logic as CentralisedAntOptimiser) --------
    private static double fitness(int k, ArrayList<WorkflowComputingAppliance> nodes, int[] sol) {
        Map<Integer, ArrayList<WorkflowComputingAppliance>> clusters = new HashMap<>();
        for (int c = 0; c < k; c++) clusters.put(c, new ArrayList<>());
        for (int i = 0; i < sol.length; i++) clusters.get(sol[i]).add(nodes.get(i));

        double f = 0.0;

        for (ArrayList<WorkflowComputingAppliance> cluster : clusters.values()) {
            if (cluster.size() < 2) return Double.MAX_VALUE; // 

            double sum = 0.0;
            for (int i = 0; i < cluster.size(); i++) {
                for (int j = i + 1; j < cluster.size(); j++) {
                    sum += CentralisedAntOptimiser.calculateHeuristic(cluster.get(i), cluster.get(j)); // 
                }
            }
            int pairs = cluster.size() * (cluster.size() - 1) / 2;
            double avg = sum / pairs;

            f += avg * Math.pow(cluster.size(), 1.5); // 
        }
        return f;
    }

    private static int[] decodeGenesFromClusters(
            HashMap<Integer, ArrayList<WorkflowComputingAppliance>> clusters,
            Map<WorkflowComputingAppliance, Integer> index,
            int n
    ) {
        int[] genes = new int[n];
        for (Map.Entry<Integer, ArrayList<WorkflowComputingAppliance>> e : clusters.entrySet()) {
            int clusterId = e.getKey();
            for (WorkflowComputingAppliance node : e.getValue()) {
                Integer i = index.get(node);
                if (i != null) genes[i] = clusterId;
            }
        }
        return genes;
    }

    private static HashMap<Integer, ArrayList<WorkflowComputingAppliance>> buildClusters(
            int k, ArrayList<WorkflowComputingAppliance> nodes, int[] sol
    ) {
        HashMap<Integer, ArrayList<WorkflowComputingAppliance>> out = new HashMap<>();
        for (int c = 0; c < k; c++) out.put(c, new ArrayList<>());
        for (int i = 0; i < sol.length; i++) out.get(sol[i]).add(nodes.get(i));
        return out;
    }

    // -------- GA operators --------
    private static class Individual {
        int[] genes;
        double fitness;

        Individual(int[] g) { genes = g; }

        static Individual random(int n, int k) {
            int[] g = new int[n];
            for (int i = 0; i < n; i++) g[i] = SeedSyncer.centralRnd.nextInt(k);
            return new Individual(g);
        }

        Individual copy() {
            int[] g2 = genes.clone();
            Individual c = new Individual(g2);
            c.fitness = fitness;
            return c;
        }
    }

    private static Individual bestOf(Individual[] pop) {
        Individual best = pop[0];
        for (int i = 1; i < pop.length; i++) if (pop[i].fitness < best.fitness) best = pop[i];
        return best;
    }

    private static Individual tournament(Individual[] pop, int t) {
        Individual best = null;
        for (int i = 0; i < t; i++) {
            Individual cand = pop[SeedSyncer.centralRnd.nextInt(pop.length)];
            if (best == null || cand.fitness < best.fitness) best = cand;
        }
        return best;
    }

    private static Individual uniformCrossover(Individual a, Individual b) {
        int n = a.genes.length;
        int[] child = new int[n];
        for (int i = 0; i < n; i++) child[i] = SeedSyncer.centralRnd.nextBoolean() ? a.genes[i] : b.genes[i];
        return new Individual(child);
    }

    private static void mutate(Individual ind, int k, double pm) {
        for (int i = 0; i < ind.genes.length; i++) {
            if (SeedSyncer.centralRnd.nextDouble() < pm) {
                ind.genes[i] = SeedSyncer.centralRnd.nextInt(k);
            }
        }
    }
    
}
