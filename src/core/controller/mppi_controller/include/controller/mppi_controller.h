/**
 * *********************************************************
 *
 * @file: mppi_controller.h
 * @brief: Contains the Model Predictive Path Integral (MPPI) local controller
 * @version: 2.0
 *
 * ********************************************************
 */
#ifndef RMP_CONTROLLER_MPPI_CONTROLLER_H_
#define RMP_CONTROLLER_MPPI_CONTROLLER_H_

#include <cstddef>
#include <random>
#include <string>
#include <vector>

#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Twist.h>
#include <nav_core/base_local_planner.h>
#include <tf2_ros/buffer.h>

#include "controller/controller.h"

namespace rmp::controller {
/**
 * @brief ROS local planner plugin for Model Predictive Path Integral control.
 */
class MPPIController : public nav_core::BaseLocalPlanner, public Controller {
public:
  MPPIController();

  MPPIController(std::string name, tf2_ros::Buffer* tf,
                 costmap_2d::Costmap2DROS* costmap_ros);

  ~MPPIController() override = default;

  void initialize(std::string name, tf2_ros::Buffer* tf,
                  costmap_2d::Costmap2DROS* costmap_ros) override;

  bool setPlan(
      const std::vector<geometry_msgs::PoseStamped>& orig_global_plan) override;

  bool computeVelocityCommands(geometry_msgs::Twist& cmd_vel) override;

  bool isGoalReached() override;

private:
  struct State {
    double x{ 0.0 };
    double y{ 0.0 };
    double yaw{ 0.0 };
  };

  struct Control {
    double v{ 0.0 };
    double omega{ 0.0 };
  };

  bool _loadParameters(ros::NodeHandle& private_nh);

  bool _transformPlan(
      const std::vector<geometry_msgs::PoseStamped>& plan,
      const std::string& target_frame,
      std::vector<geometry_msgs::PoseStamped>* transformed_plan) const;

  bool _optimizeControlSequence(
      const State& initial_state, const Control& current_control,
      const std::vector<geometry_msgs::PoseStamped>& reference_plan,
      std::size_t* valid_sample_count);

  void _predictState(State* state, const Control& control) const;

  Control _projectControl(const Control& control,
                          const Control& previous_control) const;

  void _projectControlSequence(const Control& current_control);

  std::size_t _nearestReferenceIndex(
      const State& state,
      const std::vector<geometry_msgs::PoseStamped>& reference_plan,
      std::size_t search_begin) const;

  double _referenceHeading(
      const std::vector<geometry_msgs::PoseStamped>& reference_plan,
      std::size_t index) const;

  bool _isGoalSignificantlyDifferent(
      const geometry_msgs::PoseStamped& new_goal) const;

  void _resetControlSequence();

  void _shiftControlSequence();

private:
  bool initialized_;
  bool goal_reached_;
  bool has_goal_;
  tf2_ros::Buffer* tf_;
  geometry_msgs::PoseStamped goal_pose_;

  int sample_count_;
  int prediction_steps_;
  int optimization_iterations_;
  int random_seed_;

  double noise_std_v_;
  double noise_std_omega_;
  double temperature_;
  double path_distance_weight_;
  double path_heading_weight_;
  double terminal_progress_weight_;
  double control_v_weight_;
  double control_omega_weight_;
  double control_delta_v_weight_;
  double control_delta_omega_weight_;
  double max_control_delta_v_;
  double max_control_delta_omega_;
  double goal_reset_distance_;
  double goal_reset_angle_;

  std::mt19937 random_generator_;
  std::vector<Control> nominal_controls_;
};
}  // namespace rmp::controller

#endif  // RMP_CONTROLLER_MPPI_CONTROLLER_H_
