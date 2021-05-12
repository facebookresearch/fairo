-- Copyright (c) Facebook, Inc. and its affiliates.


PLUGIN = nil

function Initialize(Plugin)
    Plugin:SetName("PointBlocks")
    Plugin:SetVersion(1)

    PLUGIN = Plugin
    cPluginManager.BindCommand("/point", "", Point, " ~ Point at specified blocks");

    LOG("Loaded plugin: point_blocks")
    return true
end

function Point(Split, Player)
    if (#Split ~= 7) then
        -- There was more or less than 6 arguments other than "/point"
        -- Sending the proper usage to player
        Player:SendMessage("Usage: /point x1 y1 z1 x2 y2 z2")
        return true
    end
    
    local CurrentWorld = Player:GetWorld()

    -- 1. Get existing block-numbers and block-metas
    block_id_arr = {}
    block_meta_arr = {}
    count = 0
    for p = Split[2],Split[5] do
        for q = Split[3],Split[6] do
            for r = Split[4],Split[7] do
                block_id_arr[count] = CurrentWorld:GetBlock(p, q, r)
                block_meta_arr[count] = CurrentWorld:GetBlockMeta(p, q, r)
                count = count + 1
            end
        end
    end

   
    -- Helper function to set blocks at given block locations
    local function Build(World)
        for p = Split[2],Split[5] do
            for q = Split[3],Split[6] do
                for r = Split[4],Split[7] do
                    -- yellow stained glass has id 95:4
                    World:SetBlock(p, q, r, 95, 4)
                end
            end
        end
    end


    -- Helper function to destropy blocks at given block locations
    local function Destroy(World)
        for p = Split[2],Split[5] do
            for q = Split[3],Split[6] do
                for r = Split[4],Split[7] do
                    World:DigBlock(p, q, r)
                end
            end
        end
    end

 
    -- Helper function to restore the original state of the blocks
    local function Restore(World)
        count = 0
        for p = Split[2],Split[5] do
            for q = Split[3],Split[6] do
                for r = Split[4],Split[7] do
                    World:SetBlock(p, q, r, block_id_arr[count], block_meta_arr[count])
                    count = count + 1
                end
            end
        end
    end

    -- 2. Perform build-and-destroy to mimic flickering
    num_flickers = 10 
    a = -1
    b = -1
    for i = 0,num_flickers do 
        a = i * 7
        b = a + 4

        CurrentWorld:ScheduleTask(a, Build)
        CurrentWorld:ScheduleTask(b, Destroy)
    end
 
    -- 3. Restore original state
    CurrentWorld:ScheduleTask(b + 1, Restore)
 
    return true  
end
