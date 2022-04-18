#include <ompl/base/ScopedState.h>
#include <ompl/base/spaces/DubinsStateSpace.h>
#include <ompl/base/spaces/ReedsSheppStateSpace.h>
#include <ompl/geometric/SimpleSetup.h>

namespace ob = ompl::base;
namespace og = ompl::geometric;

int main() {
  ob::StateSpacePtr space(std::make_shared<ob::DubinsStateSpace>());

  ob::RealVectorBounds bounds(2);
  bounds.setLow(0);
  bounds.setHigh(18);
  space->as<ob::SE2StateSpace>()->setBounds(bounds);

  og::SimpleSetup ss(space);

  const ob::SpaceInformation *si = ss.getSpaceInformation().get();
  ss.setStateValidityChecker([&](const ob::State *state) {
    const auto *s = state->as<ob::SE2StateSpace::StateType>();
    double x = s->getX();
    double y = s->getY();
    return si->satisfiesBounds(s) && (x < 5 || x > 13 || (y > 8.5 && y < 9.5));
  });

  ob::ScopedState<> start(space);
  start[0] = 1;
  start[1] = 1;
  start[2] = 0;

  ob::ScopedState<> goal(space);
  goal[0] = 17;
  goal[1] = 17;
  goal[2] = -0.99 * M_PI;

  ss.setStartAndGoalStates(start, goal);

  ss.getSpaceInformation()->setStateValidityCheckingResolution(0.005);
  ss.setup();
  ss.print();

  ob::PlannerStatus solved = ss.solve(30.0);

  if (solved) {
    std::vector<double> reals;

    std::cout << "Found solution:" << std::endl;
    ss.simplifySolution();
    og::PathGeometric path = ss.getSolutionPath();
    path.interpolate(1000);
    path.printAsMatrix(std::cout);
  } else {
    std::cout << "No solution found" << std::endl;
  }

  return 0;
}
