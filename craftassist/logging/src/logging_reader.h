// Copyright (c) Facebook, Inc. and its affiliates.


#pragma once
#include <fstream>
#include <string>
#include <tuple>
#include <vector>

#include "../../client/src/game_state.h"
#include "stream_decoder.h"

class LoggingReader {
  public:
   LoggingReader(const std::string& loggingBinPath, std::vector<std::string> mcaPaths,
                 const std::string& name);

   // Return the GameState at current tick
   GameState& currentState() { return gameState_; }

   // Return false until PLAYER_SPAWNED for the named player, then true until PLAYER_DESTROYED
   bool isPlayerInGame() { return isPlayerInGame_; }

   // Process a single hook
   void stepHook();

   // Process all hooks until (and including) the PLAYER_SPAWNED for name_
   void stepToSpawn();

   // Process all hooks until n ticks in the future
   void stepTicks(unsigned long n);

  private:
   // Save the tick for the *next* hook to be processed.
   // Returns true if valid, false if EOF.
   bool peekNextHookTick(unsigned long* nextHookTick);

   // Fields
   GameState gameState_;
   StreamDecoder log_;
   std::string name_;
   unsigned long eid_;
   long tick_ = 0;
   bool isPlayerInGame_ = false;
};
