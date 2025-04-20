package se.kth.jabeja;

import org.apache.log4j.Logger;
import se.kth.jabeja.config.Config;
import se.kth.jabeja.config.NodeSelectionPolicy;
import se.kth.jabeja.io.FileIO;
import se.kth.jabeja.rand.RandNoGenerator;

import java.io.File;
import java.io.IOException;
import java.util.*;

public class Jabeja {
  final static Logger logger = Logger.getLogger(Jabeja.class);
  private final Config config;
  private final HashMap<Integer/*id*/, Node/*neighbors*/> entireGraph;
  private final List<Integer> nodeIds;
  private int numberOfSwaps;
  private int round;
  private float T;
  private boolean resultFileCreated = false;
  private String optionAnnealing; // Add this line for Task2
  private String optionAccepyanceProb; // Add this for Task 3
  private Integer roundstoRestart; // Option to restart at initial T when min T is reached
  //-------------------------------------------------------------------
  public Jabeja(HashMap<Integer, Node> graph, Config config) {
    this.entireGraph = graph;
    this.nodeIds = new ArrayList<>(entireGraph.keySet());
    this.round = 0;
    this.numberOfSwaps = 0;
    this.config = config;
    this.T = config.getTemperature();
    // Set the initial values for Task 2
    this.optionAnnealing = config.getAnnealingOption(); // Add for Task 2
    this.roundstoRestart = config.getRoundstoRestart(); // Add for Task 2
    this.optionAccepyanceProb = config.getOptionAccepyanceProb(); // Add for Task 3
  }

  //-------------------------------------------------------------------
  public void startJabeja() throws IOException {
    for (round = 0; round < config.getRounds(); round++) {
      for (int id : entireGraph.keySet()) {
        sampleAndSwap(id);
      }

      //one cycle for all nodes have completed.
      //reduce the temperature
      saCoolDown();
      report();
    }
  }

  private void saCoolDown() {

    // Check the selected cooling strategy
    if ("linear".equals(optionAnnealing)) {
      // If linear annealing is chosen, use the linear cooling function
      linearCoolDown();
    } else if ("exp".equals(optionAnnealing)) {
      // If exponential annealing is chosen, use the exponential cooling function
      exponentialCoolDown();
    }
    // Add more cooling strategies as needed
  }

  private void linearCoolDown() {
    // Minimum temperature to prevent it from going below a certain threshold
    float tMin = 1.0f;

    // Check if the current temperature is above the minimum threshold
    if ((T > tMin) && (config.getRoundstoRestart() >= 0) && (round % config.getRoundstoRestart() == 0)){
      // Decrease temperature linearly over time by subtracting a delta value
      T -= config.getDelta();
    } else {
      // If the temperature is below the minimum threshold, check if restart is enabled
      if (config.getRoundstoRestart() >= 0 && round % config.getRoundstoRestart() == 0) {
        // Restart to the initial temperature
        T = config.getTemperature();
      } else {
        // If restart is not enabled, set it to the minimum
        T = tMin;
      }
    }
  }

  private void exponentialCoolDown() {
    // Minimum temperature to prevent it from going below a certain threshold
    float tMin = 0.0001f;

    // Check if the current temperature is above the minimum threshold
    if (T > tMin) {
      // Decrease temperature exponentially over time by multiplying it with a delta value
      T *= config.getDelta();
    } else {
      // If the temperature is below the minimum threshold, check if restart is enabled
      if (config.getRoundstoRestart() >= 0 && round % config.getRoundstoRestart() == 0) {
        // Restart to the initial temperature
        T = config.getTemperature();
      } else {
        // If restart is not enabled, set it to the minimum
        T = tMin;
      }
    }
  }

  private void sampleAndSwap(int nodeId) {
    Node partner = null;
    Node nodep = entireGraph.get(nodeId);

    if (config.getNodeSelectionPolicy() == NodeSelectionPolicy.HYBRID
            || config.getNodeSelectionPolicy() == NodeSelectionPolicy.LOCAL) {
      // TODO: swap with random neighbors
      partner = findPartner(nodeId, getNeighbors(nodep));
    }

    if (config.getNodeSelectionPolicy() == NodeSelectionPolicy.HYBRID
            || config.getNodeSelectionPolicy() == NodeSelectionPolicy.RANDOM) {
      // TODO: if local policy fails then randomly sample the entire graph
      if (partner == null)
        partner = findPartner(nodeId, getSample(nodeId));
    }

    // TODO: swap the colors
    if (partner != null) {
      int tempColor = nodep.getColor();
      nodep.setColor(partner.getColor());
      partner.setColor(tempColor);
      numberOfSwaps++; // Increment the count of performed swaps
    }
  }

  public Node findPartner(int nodeId, Integer[] nodes) {
    Node nodep = entireGraph.get(nodeId);
    Node bestPartner = null;
    double highestBenefit = 0;

    double oldValue = 0;  // Move the declaration outside the loop
    double newValue = 0;  // Move the declaration outside the loop

    for (int node : nodes) {
      Node nodeq = entireGraph.get(node);

      int dpp = getDegree(nodep, nodep.getColor());
      int dqq = getDegree(nodeq, nodeq.getColor());
      int dpq = getDegree(nodep, nodeq.getColor());
      int dqp = getDegree(nodeq, nodep.getColor());

      oldValue = Math.pow(dpp, config.getAlpha()) + Math.pow(dqq, config.getAlpha());
      newValue = Math.pow(dpq, config.getAlpha()) + Math.pow(dqp, config.getAlpha());

    // Task 3 Accepatnce Probablity
    boolean updateSolution = false;
    double currentBenefit = 0, acceptanceProb = 0;

    if ("linearAccepatnce".equals(optionAccepyanceProb)) {
      currentBenefit = highestBenefit;
      updateSolution = highestBenefit * T > oldValue;
    } else if ("expAccepatnce".equals(optionAccepyanceProb)) {
      acceptanceProb = Math.exp((highestBenefit - oldValue) / T);
      currentBenefit = acceptanceProb;
      updateSolution = acceptanceProb > Math.random() && highestBenefit != oldValue;
    } else {
      // Handle the case when optionAnnealing is neither "linear" nor "exp"
      throw new IllegalArgumentException("Invalid annealing option: " + optionAccepyanceProb);
    }

    if (newValue > oldValue && newValue > highestBenefit) {
        bestPartner = nodeq;
        highestBenefit = newValue;
      }
    }

    return bestPartner;
  }

  private int getDegree(Node node, int colorId){
    int degree = 0;
    for(int neighborId : node.getNeighbours()){
      Node neighbor = entireGraph.get(neighborId);
      if(neighbor.getColor() == colorId){
        degree++;
      }
    }
    return degree;
  }

  private Integer[] getSample(int currentNodeId) {
    int count = config.getUniformRandomSampleSize();
    int rndId;
    int size = entireGraph.size();
    ArrayList<Integer> rndIds = new ArrayList<>();

    while (true) {
      rndId = nodeIds.get(RandNoGenerator.nextInt(size));
      if (rndId != currentNodeId && !rndIds.contains(rndId)) {
        rndIds.add(rndId);
        count--;
      }

      if (count == 0)
        break;
    }

    Integer[] ids = new Integer[rndIds.size()];
    return rndIds.toArray(ids);
  }

  private Integer[] getNeighbors(Node node) {
    ArrayList<Integer> list = node.getNeighbours();
    int count = config.getRandomNeighborSampleSize();
    int rndId;
    int index;
    int size = list.size();
    ArrayList<Integer> rndIds = new ArrayList<>();

    if (size <= count)
      rndIds.addAll(list);
    else {
      while (true) {
        index = RandNoGenerator.nextInt(size);
        rndId = list.get(index);
        if (!rndIds.contains(rndId)) {
          rndIds.add(rndId);
          count--;
        }

        if (count == 0)
          break;
      }
    }

    Integer[] arr = new Integer[rndIds.size()];
    return rndIds.toArray(arr);
  }

  private void report() throws IOException {
    int grayLinks = 0;
    int migrations = 0; // number of nodes that have changed the initial color
    int size = entireGraph.size();

    for (int i : entireGraph.keySet()) {
      Node node = entireGraph.get(i);
      int nodeColor = node.getColor();
      ArrayList<Integer> nodeNeighbours = node.getNeighbours();

      if (nodeColor != node.getInitColor()) {
        migrations++;
      }

      if (nodeNeighbours != null) {
        for (int n : nodeNeighbours) {
          Node p = entireGraph.get(n);
          int pColor = p.getColor();

          if (nodeColor != pColor)
            grayLinks++;
        }
      }
    }

    int edgeCut = grayLinks / 2;

    logger.info("round: " + round +
            ", edge cut:" + edgeCut +
            ", swaps: " + numberOfSwaps +
            ", migrations: " + migrations);

    saveToFile(edgeCut, migrations);
  }

  private void saveToFile(int edgeCuts, int migrations) throws IOException {
    String delimiter = "\t\t";
    String outputFilePath;

    //output file name
    File inputFile = new File(config.getGraphFilePath());
    outputFilePath = config.getOutputDir() +
            File.separator +  // Add double File separator here
            inputFile.getName() + "_" +
            "NS" + "_" + config.getNodeSelectionPolicy() + "_" +
            "GICP" + "_" + config.getGraphInitialColorPolicy() + "_" +
            "T" + "_" + config.getTemperature() + "_" +
            "D" + "_" + config.getDelta() + "_" +
            "RNSS" + "_" + config.getRandomNeighborSampleSize() + "_" +
            "URSS" + "_" + config.getUniformRandomSampleSize() + "_" +
            "A" + "_" + config.getAlpha() + "_" +
            "R" + "_" + config.getRounds() + "_" +
            "LA" + "_" + config.getAnnealingOption() + "_" +
            "ACC_PR" + "_" + config.getOptionAccepyanceProb() + "_" +
            "RESTART_T" + "_" + config.getRoundstoRestart() + ".txt";

    if (!resultFileCreated) {
      File outputDir = new File(config.getOutputDir());
      if (!outputDir.exists()) {
        if (!outputDir.mkdir()) {
          throw new IOException("Unable to create the output directory");
        }
      }
      // create folder and result file with header
      String header = "# Migration is number of nodes that have changed color.";
      header += "\n\nRound" + delimiter + "Edge-Cut" + delimiter + "Swaps" + delimiter + "Migrations" + delimiter + "Skipped" + "\n";
      FileIO.write(header, outputFilePath);
      resultFileCreated = true;
    }

    FileIO.append(round + delimiter + (edgeCuts) + delimiter + numberOfSwaps + delimiter + migrations + "\n", outputFilePath);
  }
}
