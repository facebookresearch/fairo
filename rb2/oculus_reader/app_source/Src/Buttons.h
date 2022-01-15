//
// Created by Jedrzej on 12/15/2020.
//

#ifndef OCULUSTELEOP_BUTTONS_H
#define OCULUSTELEOP_BUTTONS_H

#include "VrApi_Input.h"
#include <string>

namespace OVRFW {

    class Buttons {

    public:
        void update_buttons(
                ovrInputStateTrackedRemote remoteInputState, const ovrHandedness controllerHand);
        std::string current_to_string(char side) const;

    private:
        ovrInputStateTrackedRemote leftRemoteInputState_;
        ovrInputStateTrackedRemote rightRemoteInputState_;
    };

}


#endif //OCULUSTELEOP_BUTTONS_H
