-- Copyright (c) Facebook, Inc. and its affiliates.


PLUGIN = nil

CHUNKS_TO_LOAD = __CHUNKS_TO_LOAD__

function Initialize(Plugin)
    Plugin:SetName("ShutdownOnLeave")

    PLUGIN = Plugin
    cPluginManager.AddHook(cPluginManager.HOOK_PLAYER_DESTROYED, OnPlayerDestroyed)

    LOG("Loaded plugin: ShutdownOnLeave.")
    return true
end

function OnPlayerDestroyed(player)
    LOGERROR("[shutdown_on_leave] Player left! Exiting now.")
    os.exit(0)
end

