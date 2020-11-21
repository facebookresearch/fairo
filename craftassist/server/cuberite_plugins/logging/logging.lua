-- Copyright (c) Facebook, Inc. and its affiliates.


PLUGIN = nil
LOGFILE = nil
VERSION_MAJOR = 0
VERSION_MINOR = 5

require("struct")

function Initialize(Plugin)
    Plugin:SetName("Logging")

    PLUGIN = Plugin
    cPluginManager.AddHook(cPluginManager.HOOK_BLOCK_SPREAD, OnBlockSpread) -- 1
    cPluginManager.AddHook(cPluginManager.HOOK_CHAT, OnChat) -- 5
    cPluginManager.AddHook(cPluginManager.HOOK_CHUNK_AVAILABLE, OnChunkAvailable) -- 6
    cPluginManager.AddHook(cPluginManager.HOOK_COLLECTING_PICKUP, OnCollectingPickup) -- 11
    cPluginManager.AddHook(cPluginManager.HOOK_EXPLODED, OnExploded) -- 19
    cPluginManager.AddHook(cPluginManager.HOOK_KILLED, OnKilled) -- 24
    cPluginManager.AddHook(cPluginManager.HOOK_PLAYER_BROKEN_BLOCK, OnPlayerBrokenBlock) -- 30
    cPluginManager.AddHook(cPluginManager.HOOK_PLAYER_DESTROYED, OnPlayerDestroyed) -- 31
    cPluginManager.AddHook(cPluginManager.HOOK_PLAYER_EATING, OnPlayerEating) -- 32
    cPluginManager.AddHook(cPluginManager.HOOK_PLAYER_FOOD_LEVEL_CHANGE, OnPlayerFoodLevelChange) -- 35
    cPluginManager.AddHook(cPluginManager.HOOK_PLAYER_MOVING, OnPlayerMoving) -- 38
    cPluginManager.AddHook(cPluginManager.HOOK_PLAYER_PLACED_BLOCK, OnPlayerPlacedBlock) -- 40
    cPluginManager.AddHook(cPluginManager.HOOK_PLAYER_SHOOTING, OnPlayerShooting) -- 44
    cPluginManager.AddHook(cPluginManager.HOOK_PLAYER_SPAWNED, OnPlayerSpawned) -- 45
    cPluginManager.AddHook(cPluginManager.HOOK_PLAYER_USED_BLOCK, OnPlayerUsedBlock) -- 47
    cPluginManager.AddHook(cPluginManager.HOOK_PLAYER_USED_ITEM, OnPlayerUsedItem) -- 48
    cPluginManager.AddHook(cPluginManager.HOOK_POST_CRAFTING, OnPostCrafting) -- 53
    -- cPluginManager.AddHook(cPluginManager.HOOK_PRE_CRAFTING, OnPreCrafting) -- 54
    cPluginManager.AddHook(cPluginManager.HOOK_PROJECTILE_HIT_BLOCK, OnProjectileHitBlock) -- 55
    cPluginManager.AddHook(cPluginManager.HOOK_PROJECTILE_HIT_ENTITY, OnProjectileHitEntity) -- 56
    cPluginManager.AddHook(cPluginManager.HOOK_SPAWNED_ENTITY, OnSpawnedEntity) -- 58
    -- cPluginManager.AddHook(cPluginManager.HOOK_SPAWNED_MONSTER, OnSpawnedMonster) -- 59
    cPluginManager.AddHook(cPluginManager.HOOK_TAKE_DAMAGE, OnTakeDamage) -- 62
    -- cPluginManager.AddHook(cPluginManager.HOOK_UPDATED_SIGN, OnUpdatedSign) -- 64
    cPluginManager.AddHook(cPluginManager.HOOK_WEATHER_CHANGED, OnWeatherChanged) -- 66
    cPluginManager.AddHook(cPluginManager.HOOK_WORLD_STARTED, OnWorldStarted) -- 68
    cPluginManager.AddHook(cPluginManager.HOOK_WORLD_TICK, OnWorldTick) -- 69
    -- cPluginManager.AddHook(cPluginManager.HOOK_MONSTER_MOVED, OnMonsterMoved) -- 70
    -- cPluginManager.AddHook(cPluginManager.HOOK_PLAYER_LOOK, OnPlayerLook) -- 71


    -- Open logfile and write version
    LOGFILE = assert(io.open("logging.bin", "wb"))
    LOG("[logging] Writing version " .. VERSION_MAJOR .. ":" .. VERSION_MINOR)
    LOGFILE:write(struct.pack('>HH', VERSION_MAJOR, VERSION_MINOR))

    LOG("[logging] Finished initialize")
    return true
end

function OnWorldStarted(world)
    -- Write sha1 of ini files
    function writeFileHash(name)
        local file = assert(io.open(name, "rb"))
        local contents = file:read("*a")
        local numBytes = string.len(contents)
        local hash = cCryptoHash.sha1(contents)
        local hashHex = cCryptoHash.sha1HexString(contents)
        LOG("[logging] Writing hash [" .. numBytes .. "b] " .. name .. " -> " .. hashHex)
        file:close()
        LOGFILE:write(hash)
    end
    LOGFILE:write(struct.pack('>BL',
        68,                 -- hook id
        world:GetWorldAge() -- world tick
    ))
    writeFileHash("settings.ini")
    writeFileHash("world/world.ini")
end

function OnDisable()
    LOG(PLUGIN:GetName() .. " is shutting down...")
    assert(LOGFILE:close())
end

function OnWorldTick(world, delta)
    if world:GetWorldAge() % 10 == 0 then
        LOGFILE:flush()
    end
end

function OnPlayerSpawned(player)
    LOGFILE:write(struct.pack('>BLLHsdddff',
        45,                              -- hook id
        player:GetWorld():GetWorldAge(), -- world tick
        player:GetUniqueID(),            -- entity id
        string.len(player:GetName()),    -- len(name)
        player:GetName(),                -- name
        player:GetPosX(),
        player:GetPosY(),
        player:GetPosZ(),
        player:GetYaw(),
        player:GetPitch()
    ))
end

function OnPlayerDestroyed(player)
    LOGFILE:write(struct.pack('>BLL',
        31,                              -- hook id
        player:GetWorld():GetWorldAge(), -- world tick
        player:GetUniqueID()             -- entity id
    ))
end

function OnPlayerMoving(player, oldpos, newpos)
    LOGFILE:write(struct.pack('>BLLdddddd',
        38,                              -- hook id
        player:GetWorld():GetWorldAge(), -- world tick
        player:GetUniqueID(),            -- entity id
        oldpos.x, oldpos.y, oldpos.z,
        newpos.x, newpos.y, newpos.z
    ))
end

function OnChunkAvailable(world, cx, cz)
    LOGFILE:write(struct.pack('>BLLL',
        6,                   -- hook id
        world:GetWorldAge(), -- world tick
        cx, cz               -- chunk x/z
    ))
end

function OnBlockSpread(world, x, y, z, source)
    LOGFILE:write(struct.pack('>BLLLLB',
        1,                   -- hook id
        world:GetWorldAge(), -- world tick
        x, y, z,             -- block x/y/z
        source
    ))
end

function OnChat(player, msg)
    local world = player:GetWorld()
    LOGFILE:write(struct.pack('>BLLHs',
        5,                    -- hook id
        world:GetWorldAge(),  -- world tick
        player:GetUniqueID(), -- entity id
        string.len(msg),      -- length of msg
        msg
    ))
end

function OnCollectingPickup(player, pickup)
    local world = player:GetWorld()
    LOGFILE:write(struct.pack('>BLLHHH',
        11,                           -- hook id
        world:GetWorldAge(),          -- world tick
        player:GetUniqueID(),         -- entity id
        pickup:GetItem().m_ItemType,  -- item id
        pickup:GetItem().m_ItemCount, -- # of items
        pickup:GetItem().m_ItemDamage -- item damage (N.B. holds meta?)
    ))
end

function OnExploded(world, size, canCauseFire, x, y, z, source, sourceData)
    LOGFILE:write(struct.pack('>BLdBLLLB',
        19,                  -- hook id
        world:GetWorldAge(), -- world tick
        size,                -- explosion size
        canCauseFire,        -- true if random air blocks catch
        x, y, z,             -- explosion center
        source               -- eExplosionSource
    ))
end

function OnKilled(victim, takeDamageInfo, deathMessage)
    local world = victim:GetWorld()
    LOGFILE:write(struct.pack('>BLL',
        24,                  -- hook id
        world:GetWorldAge(), -- world tick
        victim:GetUniqueID() -- entity id
    ))
end

function OnPlayerBrokenBlock(player, x, y, z, face, id, meta)
    local world = player:GetWorld()
    LOGFILE:write(struct.pack('>BLLLLLBBB',
        30,                   -- hook id
        world:GetWorldAge(),  -- world tick
        player:GetUniqueID(), -- entity id
        x, y, z,              -- block location
        face,                 -- block face
        id, meta              -- block id
    ))
end

function OnPlayerEating(player)
    local world = player:GetWorld()
    LOGFILE:write(struct.pack('>BLL',
        32,                  -- hook id
        world:GetWorldAge(), -- world tick
        player:GetUniqueID() -- entity id
    ))
end

function OnPlayerFoodLevelChange(player, foodLevel)
    local world = player:GetWorld()
    LOGFILE:write(struct.pack('>BLLB',
        35,                   -- hook id
        world:GetWorldAge(),  -- world tick
        player:GetUniqueID(), -- entity id
        foodLevel             -- new food level
    ))
end

function OnPlayerPlacedBlock(player, x, y, z, id, meta)
    local world = player:GetWorld()
    LOGFILE:write(struct.pack('>BLLLLLBB',
        40,                   -- hook id
        world:GetWorldAge(),  -- world tick
        player:GetUniqueID(), -- entity id
        x, y, z,              -- block location
        id, meta              -- block id
    ))
end

function OnPlayerShooting(player)
    local world = player:GetWorld()
    LOGFILE:write(struct.pack('>BLL',
        44,                  -- hook id
        world:GetWorldAge(), -- world tick
        player:GetUniqueID() -- entity id
    ))
end

function OnPlayerUsedBlock(player, x, y, z, face, cursorX, cursorY, cursorZ, id, meta)
    local world = player:GetWorld()
    LOGFILE:write(struct.pack('>BLLLLLBfffBB',
        47,                        -- hook id
        world:GetWorldAge(),       -- world tick
        player:GetUniqueID(),      -- entity id
        x, y, z,                   -- block x/y/z
        face,                      -- block face
        cursorX, cursorY, cursorZ, -- cursor crosshair on block
        id, meta                   -- block id
    ))
end

function OnPlayerUsedItem(player, x, y, z, face, cursorX, cursorY, cursorZ)
    local world = player:GetWorld()
    LOGFILE:write(struct.pack('>BLLLLLBfffH',
        48,                                 -- hook id
        world:GetWorldAge(),                -- world tick
        player:GetUniqueID(),               -- entity id
        x, y, z,                            -- block x/y/z
        face,                               -- block face
        cursorX, cursorY, cursorZ,          -- cursor crosshair on block
        player:GetEquippedItem().m_ItemType -- equipped item id
    ))
end

function OnPostCrafting(player, grid, recipe)
    local world = player:GetWorld()
    LOGFILE:write(struct.pack('>BLL',
        53,                  -- hook id
        world:GetWorldAge(), -- world tick
        player:GetUniqueID() -- entity id
    ))
    -- write crafting grid
    LOGFILE:write(struct.pack('>BB', grid:GetHeight(), grid:GetWidth()))
    for y=0,grid:GetHeight()-1 do
        for x=0,grid:GetWidth()-1 do
            local item = grid:GetItem(x, y)
            LOGFILE:write(struct.pack('>HHH',
                item.m_ItemType,
                item.m_ItemCount,
                item.m_ItemDamage))
        end
    end
    -- write recipe ingredients
    LOGFILE:write(struct.pack('>BB',
        recipe:GetIngredientsHeight(),
        recipe:GetIngredientsWidth()))
    for y=0,recipe:GetIngredientsHeight()-1 do
        for x=0,recipe:GetIngredientsWidth()-1 do
            local item = recipe:GetIngredient(x, y)
            LOGFILE:write(struct.pack('>HHH',
                item.m_ItemType,
                item.m_ItemCount,
                item.m_ItemDamage))
        end
    end
    -- write recipe result
    local result = recipe:GetResult()
    LOGFILE:write(struct.pack('>HHH',
        result.m_ItemType,
        result.m_ItemCount,
        result.m_ItemDamage))
end

function OnProjectileHitBlock(projectile, x, y, z, face, hitPos)
    local world = projectile:GetWorld()
    LOGFILE:write(struct.pack('>BLLLLLBddd',
        55,                          -- hook id
        world:GetWorldAge(),         -- world tick
        projectile:GetUniqueID(),    -- entity id
        x, y, z, face,               -- block pos, face
        hitPos.x, hitPos.y, hitPos.z -- exact pos where projectile hit
    ))
end

function OnProjectileHitEntity(projectile, target)
    local world = projectile:GetWorld()
    LOGFILE:write(struct.pack('>BLLL',
        56,                       -- hook id
        world:GetWorldAge(),      -- world tick
        projectile:GetUniqueID(), -- projectile entity id
        target:GetUniqueID()      -- target entity id
    ))
end

function OnSpawnedEntity(world, entity)
    LOGFILE:write(struct.pack('>BLLBdddff',
        58,                     -- hook id
        world:GetWorldAge(),    -- world tick
        entity:GetUniqueID(),   -- entity id
        entity:GetEntityType(), -- eEntityType
        entity:GetPosX(),       -- spawn position
        entity:GetPosY(),
        entity:GetPosZ(),
        entity:GetYaw(),        -- look
        entity:GetPitch()
    ))
    if entity:IsMob() then
        LOGFILE:write(struct.pack('>B', entity:GetMobType()))
    end
end

-- This hook is unreliably called, so we save the info we need in
-- OnSpawnedEntity instead. See
-- https://github.com/cuberite/cuberite/issues/4193
--
-- function OnSpawnedMonster(world, monster)
--     LOGFILE:write(struct.pack('>BLLBBdddff',
--         59,                     -- hook id
--         world:GetWorldAge(),    -- world tick
--         monster:GetUniqueID(),  -- entity id
--         monster:GetMobFamily(), -- eFamily
--         monster:GetMobType(),   -- eMonsterType
--         monster:GetPosX(),      -- spawn position
--         monster:GetPosY(),
--         monster:GetPosZ(),
--         monster:GetYaw(),       -- look
--         monster:GetPitch()
--     ))
-- end

function OnTakeDamage(entity, tdi)
    local world = entity:GetWorld()
    LOGFILE:write(struct.pack('>BLLBddddd',
        62,                       -- hook id
        world:GetWorldAge(),      -- world tick
        entity:GetUniqueID(),     -- entity id
        tdi.DamageType,           -- eDamageType
        tdi.FinalDamage,          -- damage to be applied to receiver
        tdi.RawDamage,            -- attack damage excluding armor
        tdi.Knockback.x,          -- knockback movement
        tdi.Knockback.y,
        tdi.Knockback.z
    ))
    -- attacker entity id only if damage type is dtAttack
    if tdi.DamageType == dtAttack then
        LOGFILE:write(struct.pack('>L', tdi.Attacker:GetUniqueID()))
    end
end

function OnWeatherChanged(world)
    LOGFILE:write(struct.pack('>BLB',
        66,                   -- hook id
        world:GetWorldAge(),  -- world tick
        world:GetWeather()    -- eWeather
    ))
end

function OnMonsterMoved(monster, oldpos, newpos)
    local world = monster:GetWorld()
    LOGFILE:write(struct.pack('>BLLdddff',
        70,                    -- hook id
        world:GetWorldAge(),   -- world tick
        monster:GetUniqueID(), -- entity id
        newpos.x,
        newpos.y,
        newpos.z,
        monster:GetYaw(),
        monster:GetPitch()
    ))
end

function OnPlayerLook(player, yaw, pitch) -- 71
    local world = player:GetWorld()
    LOGFILE:write(struct.pack('>BLLff',
        71,                    -- hook id
        world:GetWorldAge(),   -- world tick
        player:GetUniqueID(),  -- entity id
        yaw,
        pitch
    ))
end
