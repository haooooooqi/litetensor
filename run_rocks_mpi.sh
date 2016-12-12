#!/bin/bash

# Max allowed nodes
MAX_ALLOWED_NODES=4

# Each node will use up to 16 cores
TOTAL_PROCESSORS_PER_NODE=32

# 10 minutes time limit
WALLTIME=30

# Ensure 3 arguments for nodes, processors per node, and input file.
if [ $# -ne 3 ]; then
  echo "Usage: $(basename $0) nodes processors_per_node input_file"
  exit $E_BADARGS
fi

# Get command line arguments.
NODES=$1
PROCESSORS_PER_NODE=$2
INPUT_FILE=$3

# Validate arguments.
if [ $NODES -le 0 ] || [ $NODES -gt $MAX_ALLOWED_NODES ]; then
    echo "ERROR: Only $MAX_ALLOWED_NODES nodes allowed."
    exit $E_BADARGS
fi
if [ $PROCESSORS_PER_NODE -le 0 ] || [ $PROCESSORS_PER_NODE -gt $TOTAL_PROCESSORS_PER_NODE ]; then
    echo "ERROR: Each node only has $TOTAL_PROCESSORS_PER_NODE cores."
    exit $E_BADARGS
fi
if [ ! -f $INPUT_FILE ]; then
    echo "ERROR: Input file $INPUT_FILE does not exist."
    exit $E_BADARGS
fi
if [ ! -f "./cpd" ]; then
    echo "ERROR: ./wire_route program does not exist."
    exit $E_BADARGS
fi

# Submit the job.  No need to modify this.
qsub -l walltime=0:$WALLTIME:00,nodes=$NODES:ppn=$TOTAL_PROCESSORS_PER_NODE -F "$NODES $PROCESSORS_PER_NODE $INPUT_FILE" rocks.qsub
