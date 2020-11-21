-- Copyright (c) Facebook, Inc. and its affiliates.


PLUGIN = nil

function Initialize(Plugin)
    Plugin:SetName("debug")
    Plugin:SetVersion(1)

    PLUGIN = Plugin
    cPluginManager.AddHook(cPluginManager.HOOK_PLAYER_USED_ITEM, OnPlayerUsedItem)

    LOG("Loaded plugin: debug")
    return true
end

function OnPlayerUsedItem(player, bx, by, bz, face, cx, cy, cz)
    LOG(bx .. " " .. by .. " " .. bz .. " " .. face .. " " .. cx .. " " .. cy .. " " .. cz)
end
