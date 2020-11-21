-- Copyright (c) Facebook, Inc. and its affiliates.


PLUGIN = nil

function Initialize(Plugin)
    Plugin:SetName("chatlog")
    Plugin:SetVersion(1)

    PLUGIN = Plugin
    cPluginManager.AddHook(cPluginManager.HOOK_CHAT, OnChat)

    LOG("Loaded plugin: chatlog")
    return true
end

function OnChat(player, chat)
    LOG("[chatlog] " .. player:GetName() .. ": " .. chat)
end
