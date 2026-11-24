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

#include <geometry_msgs/Point.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Twist.h>
#include <nav_core/base_local_planner.h>
#include <nav_msgs/Path.h>
#include <ros/ros.h>
#include <tf2_ros/buffer.h>
#include <visualization_msgs/Marker.h>

#include "controller/controller.h"

namespace base_local_planner {
class CostmapModel;
}

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

  enum class CollisionCheckResult {
    kFree,
    kCollision,
    kOutsideMap
  };

  bool _loadParameters(ros::NodeHandle& private_nh);

  bool _transformPlan(
      const std::vector<geometry_msgs::PoseStamped>& plan,
      const std::string& target_frame,
      std::vector<geometry_msgs::PoseStamped>* transformed_plan) const;

  bool _optimizeControlSequence(
      const State& initial_state, const Control& current_control,
      const std::vector<geometry_msgs::PoseStamped>& reference_plan,
      base_local_planner::CostmapModel& collision_model,
      const std::vector<geometry_msgs::Point>& footprint,
      double inscribed_radius, double circumscribed_radius,
      std::size_t* valid_sample_count,
      std::size_t* collision_rejection_count,
      std::size_t* safety_rejection_count,
      std::size_t* unverified_sample_count,
      std::vector<std::vector<State>>* sampled_trajectories,
      std::vector<std::vector<State>>* rejected_trajectories);

  CollisionCheckResult _checkStateCollision(
      const State& state,
      base_local_planner::CostmapModel& collision_model,
      const std::vector<geometry_msgs::Point>& footprint,
      double inscribed_radius, double circumscribed_radius) const;

  CollisionCheckResult _checkMotionCollision(
      const State& from, const State& to,
      base_local_planner::CostmapModel& collision_model,
      const std::vector<geometry_msgs::Point>& footprint,
      double inscribed_radius, double circumscribed_radius) const;

  bool _isControlSequenceCollisionFree(
      const State& initial_state, const std::vector<Control>& controls,
      base_local_planner::CostmapModel& collision_model,
      const std::vector<geometry_msgs::Point>& footprint,
      double inscribed_radius, double circumscribed_radius) const;

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

  bool _visualizationDue(const ros::Time& stamp) const;

  void _publishVisualization(
      const State& initial_state, const Control& current_control,
      const std::vector<std::vector<State>>& sampled_trajectories,
      const std::vector<std::vector<State>>& rejected_trajectories,
      const std::string& frame_id, const ros::Time& stamp);

  void _publishTrajectoryMarker(
      const std::vector<std::vector<State>>& trajectories,
      const std::string& marker_namespace, const std::string& frame_id,
      const ros::Time& stamp, float red, float green, float blue, float alpha,
      ros::Publisher* publisher) const;

  nav_msgs::Path _buildOptimizedPath(
      const State& initial_state, const Control& current_control,
      const std::string& frame_id, const ros::Time& stamp) const;

  void _clearVisualization();

private:
  bool initialized_;
  bool goal_reached_;
  bool has_goal_;
  bool footprint_logged_;
  bool visualization_enabled_;
  bool visualization_active_;
  tf2_ros::Buffer* tf_;
  geometry_msgs::PoseStamped goal_pose_;
  ros::Publisher sampled_trajectories_publisher_;
  ros::Publisher rejected_trajectories_publisher_;
  ros::Publisher optimized_trajectory_publisher_;
  ros::Time last_visualization_time_;
  std::string visualization_frame_;

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
  double collision_check_resolution_;
  bool treat_unknown_as_collision_;
  double collision_safety_horizon_;
  double visualization_frequency_;
  int visualization_sample_count_;
  int visualization_rejected_count_;
  double goal_reset_distance_;
  double goal_reset_angle_;

  std::mt19937 random_generator_;
  std::vector<Control> nominal_controls_;
};
}  // namespace rmp::controller

#endif  // RMP_CONTROLLER_MPPI_CONTROLLER_H_
