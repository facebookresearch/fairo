-- Copyright (c) Facebook, Inc. and its affiliates.


PLUGIN = nil

function Initialize(Plugin)
    Plugin:SetName("spawn_position")
    Plugin:SetVersion(1)

    PLUGIN = Plugin
    cPluginManager.AddHook(cPluginManager.HOOK_WORLD_STARTED, OnWorldStarted)

    LOG("Loaded plugin: spawn_position")
    return true
end

function OnWorldStarted(world)
    file = io.open("spawn.txt", "w")
    file:write(world:GetSpawnX() .. "\n" .. world:GetSpawnY() .. "\n" .. world:GetSpawnZ())
    file:close()
end
