-- Copyright (c) Facebook, Inc. and its affiliates.


PLUGIN = nil

CHUNKS_TO_LOAD = __CHUNKS_TO_LOAD__

hasPlayerJoined = false
-- One second is 20 ticks, allow 15 minutes inactive
MAX_INACTIVE_TICK = 20 * 60 * 15


function Initialize(Plugin)
    Plugin:SetName("ShutdownIfNoPlayerJoin")

    PLUGIN = Plugin
    hasPlayerJoined = false
    local world = cRoot:Get():GetDefaultWorld()

    cPluginManager.AddHook(cPluginManager.HOOK_PLAYER_JOINED, OnPlayerJoined)

    local callback;
    callback = function(world)
      if hasPlayerJoined == false then
        LOGERROR("[shutdown_if_no_player_join] No player has joined the server in a while! Exiting now.")
        os.exit(0)
      end
    end

    world:ScheduleTask(MAX_INACTIVE_TICK, callback)

    LOG("Loaded plugin: ShutdownIfNoPlayerJoin.")
    return true
end

function OnPlayerJoined(player)
    hasPlayerJoined = true
    LOG("Player joined the game.")
end
