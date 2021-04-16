-- Copyright (c) Facebook, Inc. and its affiliates.


PLUGIN = nil

function Initialize(Plugin)
    Plugin:SetName("starting_inventory")
    Plugin:SetVersion(1)

    PLUGIN = Plugin
    cPluginManager.AddHook(cPluginManager.HOOK_PLAYER_SPAWNED, OnPlayerSpawned)
    cPluginManager.AddHook(cPluginManager.HOOK_WORLD_STARTED, OnWorldStarted)

    LOG("Loaded plugin: starting_inventory")
    return true
end

function OnPlayerSpawned(player)
    local items = cItems()
    items:Add(cItem(E_ITEM_STICK, 2))
    items:Add(cItem(E_BLOCK_PLANKS, 3))

    player:GetInventory():AddItems(items, true)
end

function OnWorldStarted(world)
    world:SetBlock(
        world:GetSpawnX(),
        world:GetSpawnY(),
        world:GetSpawnZ() + 2,
        E_BLOCK_CRAFTING_TABLE,
        0)
end
