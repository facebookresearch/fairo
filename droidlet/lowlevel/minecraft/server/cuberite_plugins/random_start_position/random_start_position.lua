-- Copyright (c) Facebook, Inc. and its affiliates.


PLUGIN = nil

function Initialize(Plugin)
    Plugin:SetName("RandomStartPosition")
    Plugin:SetVersion(1)

    PLUGIN = Plugin
    cPluginManager.AddHook(cPluginManager.HOOK_PLAYER_SPAWNED, OnPlayerSpawned)

    LOG("Loaded plugin: RandomStartPosition")
    math.randomseed(0)
    return true
end

function OnPlayerSpawned(player)
    x = math.random(-1e4,1e4)
    z = math.random(-1e4,1e4)
    y = player:GetWorld():GetHeight(x, z)
    player:TeleportToCoords(x, y + 1, z)
end
