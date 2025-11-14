#!/bin/bash
# Master script to submit PBS jobs for training
# Usage: ./submit_jobs.sh [task] [model] [--all]

SCRIPT_DIR="pbs_scripts_expanded"

# Color codes for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to submit a single job
submit_job() {
    local script=$1
    if [ -f "$script" ]; then
        # Get absolute path for qsub
        script_abs=$(realpath "$script")
        job_id=$(qsub "$script_abs" 2>&1)
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✓${NC} Submitted: $(basename $script) -> $job_id"
        else
            echo -e "${YELLOW}✗${NC} Failed: $(basename $script)"
            echo -e "      Error: $job_id"
        fi
        sleep 0.5  # Small delay between submissions
    else
        echo -e "${YELLOW}✗${NC} Not found: $script"
    fi
}

# Function to show usage
show_usage() {
    cat << EOF
Usage: ./submit_jobs.sh [OPTIONS]

Submit PBS training jobs to the cluster.

Options:
  --all                Submit all jobs (296 total)
  --task TASK          Submit all jobs for a specific task
                       (worldfloods, phisatnet_clouds, roads, building,
                        lpl_burned_area, fire, anomaly_detection)
  --model MODEL        Submit all jobs for a specific model
                       (caco, seco, phisatnet, dino, moco, gassl,
                        geoaware, prithvi, satmae, uniphi, vit)
  --nshot NSHOT        Submit all jobs for a specific n_shot value
                       (50, 100, 500, 1000, 5000)
  --script SCRIPT      Submit a specific script by name
  --count              Count available scripts by task
  --list               List sample scripts (first 5 per task)
  --status             Check status of user's jobs
  -h, --help           Show this help message

Examples:
  ./submit_jobs.sh --task worldfloods       # Submit all worldfloods jobs (60)
  ./submit_jobs.sh --model caco             # Submit all caco model jobs (30)
  ./submit_jobs.sh --nshot 50               # Submit all n_shot=50 jobs (59)
  ./submit_jobs.sh --script worldfloods_caco_nshot50_frozen.sh
  ./submit_jobs.sh --count                  # Count scripts by task
  ./submit_jobs.sh --all                    # Submit everything (296 jobs!)
  ./submit_jobs.sh --status                 # Check job status

EOF
}

# Function to count scripts
count_scripts() {
    echo -e "${BLUE}Script Count by Task:${NC}"
    echo ""
    local total=0
    for task in worldfloods phisatnet_clouds roads building lpl_burned_area fire anomaly_detection; do
        count=$(ls -1 $SCRIPT_DIR/${task}_*.sh 2>/dev/null | wc -l)
        if [ $count -gt 0 ]; then
            echo -e "  ${GREEN}$task:${NC} $count scripts"
            total=$((total + count))
        fi
    done
    echo ""
    echo -e "  ${BLUE}Total: $total scripts${NC}"
}

# Function to list sample scripts
list_scripts() {
    echo -e "${BLUE}Sample PBS Scripts (first 5 per task):${NC}"
    echo ""
    for task in worldfloods phisatnet_clouds roads building lpl_burned_area fire anomaly_detection; do
        count=$(ls -1 $SCRIPT_DIR/${task}_*.sh 2>/dev/null | wc -l)
        if [ $count -gt 0 ]; then
            echo -e "${GREEN}$task:${NC} $count scripts total"
            ls -1 $SCRIPT_DIR/${task}_*.sh 2>/dev/null | head -5 | sed 's|.*/||' | sed 's/^/  - /'
            if [ $count -gt 5 ]; then
                echo "  ... and $((count - 5)) more"
            fi
            echo ""
        fi
    done
}

# Function to check job status
check_status() {
    echo -e "${BLUE}Current Job Status:${NC}"
    qstat -u $USER
}

# Parse command line arguments
if [ $# -eq 0 ]; then
    show_usage
    exit 0
fi

case "$1" in
    --all)
        echo -e "${BLUE}Submitting all PBS jobs...${NC}"
        count=0
        for script in $SCRIPT_DIR/*.sh; do
            if [[ $(basename "$script") != "README.md" ]]; then
                submit_job "$script"
                ((count++))
            fi
        done
        echo -e "\n${GREEN}Submitted $count jobs${NC}"
        echo "Check status with: qstat -u \$USER"
        ;;
    
    --task)
        if [ -z "$2" ]; then
            echo "Error: Task name required"
            show_usage
            exit 1
        fi
        task=$2
        echo -e "${BLUE}Submitting jobs for task: $task${NC}"
        count=0
        for script in $SCRIPT_DIR/${task}_*.sh; do
            if [ -f "$script" ]; then
                submit_job "$script"
                ((count++))
            fi
        done
        if [ $count -eq 0 ]; then
            echo "No scripts found for task: $task"
        else
            echo -e "\n${GREEN}Submitted $count jobs for $task${NC}"
        fi
        ;;
    
    --model)
        if [ -z "$2" ]; then
            echo "Error: Model name required"
            show_usage
            exit 1
        fi
        model=$2
        echo -e "${BLUE}Submitting jobs for model: $model${NC}"
        count=0
        for script in $SCRIPT_DIR/*_${model}_*.sh; do
            if [ -f "$script" ]; then
                submit_job "$script"
                ((count++))
            fi
        done
        if [ $count -eq 0 ]; then
            echo "No scripts found for model: $model"
        else
            echo -e "\n${GREEN}Submitted $count jobs for $model${NC}"
        fi
        ;;
    
    --nshot)
        if [ -z "$2" ]; then
            echo "Error: n_shot value required"
            show_usage
            exit 1
        fi
        nshot=$2
        echo -e "${BLUE}Submitting jobs for n_shot=$nshot${NC}"
        count=0
        for script in $SCRIPT_DIR/*_nshot${nshot}_*.sh; do
            if [ -f "$script" ]; then
                submit_job "$script"
                ((count++))
            fi
        done
        if [ $count -eq 0 ]; then
            echo "No scripts found for n_shot=$nshot"
        else
            echo -e "\n${GREEN}Submitted $count jobs for n_shot=$nshot${NC}"
        fi
        ;;
    
    --count)
        count_scripts
        ;;
    
    --script)
        if [ -z "$2" ]; then
            echo "Error: Script name required"
            show_usage
            exit 1
        fi
        script=$SCRIPT_DIR/$2
        submit_job "$script"
        ;;
    
    --list)
        list_scripts
        ;;
    
    --status)
        check_status
        ;;
    
    -h|--help)
        show_usage
        ;;
    
    *)
        echo "Unknown option: $1"
        show_usage
        exit 1
        ;;
esac
