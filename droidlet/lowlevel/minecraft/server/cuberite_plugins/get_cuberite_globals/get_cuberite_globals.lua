-- Copyright (c) Facebook, Inc. and its affiliates.


PLUGIN = nil

function Initialize(Plugin)
    Plugin:SetName("get_cuberite_globals")
    Plugin:SetVersion(1)

    function inner()
        -- print mob <id> <name>
        for e = 1, 255 do
            local name
            local good = pcall(function() name = cMonster:MobTypeToVanillaName(e) end)
            if good and name ~= "" then
                LOG("cuberite_global mob " .. e .. " " .. name)
            end
        end
    end

    if pcall(inner) then
        os.exit(0)
    else
        os.exit(1)
    end
end
