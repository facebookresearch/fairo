-- Copyright (c) Facebook, Inc. and its affiliates.


PLUGIN = nil

AGREED = {}

function Initialize(Plugin)
    Plugin:SetName("agreement")
    Plugin:SetVersion(1)

    PLUGIN = Plugin
    cPluginManager.AddHook(cPluginManager.HOOK_PLAYER_JOINED, OnPlayerJoined)
    cPluginManager.AddHook(cPluginManager.HOOK_PLAYER_MOVING, OnPlayerMoving)
    cPluginManager.AddHook(cPluginManager.HOOK_CHAT, OnChat)

    LOG("Loaded plugin: agreement")
    return true
end

function OnPlayerJoined(player)
    player:SendMessageInfo("Your actions are being recorded. Please read the Participation Agreement at https://fb.quip.com/yNDtAOb4ygXu")
    player:SendMessageInfo("To unlock play, press \"t\" to open chat, type \"agree\", and press Enter")
end

function OnPlayerMoving(player, oldpos, newpos)
    if AGREED[player:GetName()] then
    else
        LOG("Freeze " .. player:GetName())
        player:Freeze(newpos)
    end
end

function OnChat(player, chat)
    if chat == "agree" then
        player:SendMessageInfo("You have agreed to the Participation Agreement, and you can now move around.")
        AGREED[player:GetName()] = true
        player:Unfreeze()
    end
end
