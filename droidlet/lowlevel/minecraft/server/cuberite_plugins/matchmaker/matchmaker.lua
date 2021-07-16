-- Copyright (c) Facebook, Inc. and its affiliates.


PLUGIN = nil

SPAWN_SEPARATION = 2000

NEEDS_MATCH = nil
NEXT_SPAWN  = 0
MATCHES = {}

function Initialize(Plugin)
    Plugin:SetName("matchmaker")
    Plugin:SetVersion(1)

    PLUGIN = Plugin
    cPluginManager.AddHook(cPluginManager.HOOK_PLAYER_SPAWNED, OnPlayerSpawned)
    cPluginManager.AddHook(cPluginManager.HOOK_PLAYER_DESTROYED, OnPlayerDestroyed)
    cPluginManager.AddHook(cPluginManager.HOOK_CHAT, OnChat)

    LOG("Loaded plugin: matchmaker")
    return true
end

function OnPlayerSpawned(player)
    if NEEDS_MATCH == nil then
        teleportToSpawn(player, NEXT_SPAWN)
        NEXT_SPAWN = NEXT_SPAWN + SPAWN_SEPARATION
        NEEDS_MATCH = player:GetName()
    else
        teleportToPlayer(player, NEEDS_MATCH)
        MATCHES[NEEDS_MATCH] = player:GetName()
        MATCHES[player:GetName()] = NEEDS_MATCH
        NEEDS_MATCH = nil
    end
end

function OnPlayerDestroyed(player)
    if player:GetName() == NEEDS_MATCH then
        NEEDS_MATCH = nil
    end
end

function OnChat(player, msg)
    -- Mute everybody except your match
    local fromName = player:GetName()
    toName = MATCHES[fromName]
    sendPrivateMessage(player:GetWorld(), fromName, fromName, msg)  -- see own message
    if toName ~= nil then
        LOG("[matchmaker] chat " .. fromName .. " -> " .. toName .. ": " .. msg)
        sendPrivateMessage(player:GetWorld(), fromName, toName, msg)
    else
        LOG("[matchmaker] chat w/ no match, " .. fromName .. ": " .. msg)
    end
    return true  -- block broadcast
end

function teleportToSpawn(player, spawn)
    LOG("[matchmaker] Teleporting " .. player:GetName() .. " to x=" .. spawn)
    local height = player:GetWorld():GetHeight(spawn, 0)
    player:TeleportToCoords(spawn + 0.5, height + 1, 0.5)
end

function teleportToPlayer(player, toPlayerName)
    LOG("[matchmaker] Teleporting " .. player:GetName() .. " to " .. toPlayerName)
    player:GetWorld():DoWithPlayer(toPlayerName, function(toPlayer)
        player:TeleportToEntity(toPlayer)
    end)
end

function sendPrivateMessage(world, fromName, toName, msg)
    world:DoWithPlayer(toName, function(toPlayer)
        toPlayer:SendMessagePrivateMsg(msg, fromName)
    end)    
end
