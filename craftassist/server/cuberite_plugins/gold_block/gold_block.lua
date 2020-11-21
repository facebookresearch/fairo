-- Copyright (c) Facebook, Inc. and its affiliates.


PLUGIN = nil

function Initialize(Plugin)
    Plugin:SetName("GoldBlock")
    Plugin:SetVersion(1)

    PLUGIN = Plugin
    cPluginManager.AddHook(cPluginManager.HOOK_CHUNK_GENERATED, OnChunkGenerated)

    LOG("Loaded plugin: GoldBlock")
    return true
end

function OnChunkGenerated(world, cx, cz, chunkDesc)
    if (cx ~= 0 or cz ~= 0) then
        return false
    end

    -- Place gold block (id=41) at (10, 10)
    local x = 10
    local z = 10
    local y = chunkDesc:GetHeight(x, z)
    chunkDesc:SetBlockType(x, y, z, 41)
end
