// ROS stuff
#include <rclcpp/rclcpp.hpp>
#include <rcutils/logging_macros.h>

// Darknet messages
#include <perception_interfaces/msg/bounding_box.hpp>
#include <perception_interfaces/msg/bounding_boxes.hpp>

// Monitor
#include <monitoring_interfaces/msg/satisfaction.hpp>
#include <monitoring_interfaces/msg/satisfaction_array.hpp>

#include <percemon/percemon.hpp>

#include <std_msgs/msg/header.hpp>

#include <array>
#include <memory>
#include <random>
#include <string>
#include <string_view>
#include <vector>
#include <functional>

namespace {

constexpr double IMG_WIDTH  = 1224;
constexpr double IMG_HEIGHT = 370;

percemon::Expr get_phi1() {
  using namespace percemon;

  auto id1 = Var_id{"1"};
  auto id2 = Var_id{"2"};
  Expr phi = Forall({id1})->dot(
      Expr{Previous(Const{true})} >>
      Previous(Exists({id2})->dot((id1 == id2) & Expr{Class(id1) == Class(id2)})));

  return phi;
}

percemon::Expr get_phi2() {
  using namespace percemon;

  auto id1 = Var_id{"1"};
  auto id2 = Var_id{"2"};
  auto x   = Var_x{"1"};
  auto f   = Var_f{"1"};

  Expr phi1 = And({1 <= f - C_FRAME{}, f - C_FRAME{} <= 2});
  Expr phi2 = Exists({id2})->dot((id1 == id2) & Expr{Class(id1) == Class(id2)});

  Expr phi = Forall({id1})->at({x, f})->dot(Always(phi1 >> phi2));
  return phi;
}
}  // namespace

namespace perception_monitor {

struct Monitor : public rclcpp::Node {
 private:
  /// Bounding boxes subscriber
  rclcpp::Subscription<perception_interfaces::msg::BoundingBoxes>::SharedPtr m_bbox_sub;

  /// Satisfaction msg publisher
  rclcpp::Publisher<monitoring_interfaces::msg::SatisfactionArray>::SharedPtr m_sat_pub;

  /// Monitor
  percemon::monitoring::OnlineMonitor m_monitor1;
  percemon::monitoring::OnlineMonitor m_monitor2;

  size_t m_frame_num = 0;

 public:
  void handle_bbox(const perception_interfaces::msg::BoundingBoxes::SharedPtr msg) {
    auto frame      = percemon::datastream::Frame{};
    frame.timestamp = rclcpp::Time{msg->header.stamp}.seconds();
    frame.frame_num = this->m_frame_num++;
    frame.size_x    = msg->width;
    frame.size_y    = msg->height;
    for (const auto& bbox : msg->bounding_boxes) {
      std::string id    = bbox.class_label;
      auto obj          = percemon::datastream::Object{};
      obj.object_class  = bbox.class_id;
      obj.probability   = bbox.probability;
      obj.bbox.xmax     = bbox.xmax;
      obj.bbox.xmin     = bbox.xmin;
      obj.bbox.ymin     = bbox.ymin;
      obj.bbox.ymax     = bbox.ymax;
      frame.objects[id] = obj;
    }

    this->m_monitor1.add_frame(frame);
    this->m_monitor2.add_frame(frame);

    auto ret     = monitoring_interfaces::msg::SatisfactionArray{};
    auto header  = msg->header;
    // header.stamp = this->get_clock()->now();
    ret.header   = header;
    auto values  = std::vector<bool>{};
    values.push_back(m_monitor1.eval() >= 0);
    values.push_back(m_monitor2.eval() >= 0);
    ret.values = std::move(values);

    this->m_sat_pub->publish(ret);
  }

  Monitor() :
      rclcpp::Node("perception_monitor"),
      m_monitor1{get_phi1(), 10.0, IMG_WIDTH, IMG_HEIGHT},
      m_monitor2{get_phi2(), 10.0, IMG_WIDTH, IMG_HEIGHT} {
    RCLCPP_INFO(get_logger(), "*****************************");
    RCLCPP_INFO(get_logger(), " Perception Monitoring Node ");
    RCLCPP_INFO(get_logger(), "*****************************");
    RCLCPP_INFO(get_logger(), " * namespace: %s", get_namespace());
    RCLCPP_INFO(get_logger(), " * node name: %s", get_name());
    RCLCPP_INFO(get_logger(), "*****************************");

    this->m_bbox_sub =
        this->create_subscription<perception_interfaces::msg::BoundingBoxes>(
            "/object_detector/detections",
            10,
            std::bind(&Monitor::handle_bbox, this, std::placeholders::_1));

    this->m_sat_pub =
        this->create_publisher<monitoring_interfaces::msg::SatisfactionArray>(
            "/perception_monitor/satisfaction", 10);
  }
};

}  // namespace perception_monitor

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);

  auto node = std::make_shared<perception_monitor::Monitor>();

  rclcpp::spin(node);
  rclcpp::shutdown();
}
