/**
 * *********************************************************
 *
 * @file: mppi_controller.h
 * @brief: Contains the Model Predictive Path Integral (MPPI) local controller
 * @version: 1.0
 *
 * ********************************************************
 */
#ifndef RMP_CONTROLLER_MPPI_CONTROLLER_H_
#define RMP_CONTROLLER_MPPI_CONTROLLER_H_

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
 *
 * The first implementation stage only validates plugin integration and safely
 * commands zero velocity. Sampling and optimization are added in later stages.
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
  bool initialized_;
  bool goal_reached_;
  tf2_ros::Buffer* tf_;
};
}  // namespace rmp::controller

#endif  // RMP_CONTROLLER_MPPI_CONTROLLER_H_
