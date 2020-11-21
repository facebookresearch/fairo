-- Copyright (c) Facebook, Inc. and its affiliates.


-- The purpose of this plugin is to recover the initial block map of the world.
-- It ensures that specific chunks have been loaded while preventing block
-- changes, then disconnects the server, leaving the region .mca files for
-- analysis.

PLUGIN = nil

CHUNKS_TO_LOAD = __CHUNKS_TO_LOAD__

function Initialize(Plugin)
    Plugin:SetName("RecoverInitial")

    PLUGIN = Plugin
    cPluginManager.AddHook(cPluginManager.HOOK_LOGIN, OnLogin)
    cPluginManager.AddHook(cPluginManager.HOOK_BLOCK_SPREAD, OnBlockSpread)
    cPluginManager.AddHook(cPluginManager.HOOK_EXPLODING, OnExploding)
    cPluginManager.AddHook(cPluginManager.HOOK_CHUNK_AVAILABLE, OnChunkAvailable)
    cPluginManager.AddHook(cPluginManager.HOOK_WORLD_STARTED, OnWorldStarted)

    LOG("Loaded plugin: RecoverInitial. Recovering " .. #CHUNKS_TO_LOAD .. " chunks.")
    return true
end

function OnLogin(client, protocolVersion, username)
    LOGERROR("[recover_initial]" .. username .. " logged in, and they shouldn't have. Exiting...")
    assert(false)
end

function OnBlockSpread(world, x, y, z, source)
    LOG("[recover_initial] Blocking block spread at " .. x .. " " .. y .. " " .. z)
    return true
end

function OnExploding(world, explosionSize, canCauseFire, x, y, z, source, sourceData)
    LOG("[recover_initial] Blocking explosion at " .. x .. " " .. y .. " " .. z)
    return true
end

function OnChunkAvailable(world, cx, cz)
    -- remove {cx, cz} from CHUNKS_TO_LOAD
    for i=1,#CHUNKS_TO_LOAD do
        local c = CHUNKS_TO_LOAD[i]
        if cx == c[1] and cz == c[2] then
            table.remove(CHUNKS_TO_LOAD, i)
            break
        end
    end
    if #CHUNKS_TO_LOAD == 0 then
        LOG("[recover_initial] All chunks loaded. Exiting..")
        os.exit(0)
    end
end

function OnWorldStarted(world)
    -- force all remaining unloaded chunks to load
    for i=1,#CHUNKS_TO_LOAD do
        local c = CHUNKS_TO_LOAD[i]
        world:GenerateChunk(c[1], c[2])
    end
end

