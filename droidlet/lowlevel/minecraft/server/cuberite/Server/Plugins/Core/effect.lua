function HandleEffectCommand(Split, Player)
	if Split[4] == nil then
		Split[4] = 30
	end
	if Split[5] == nil then
		Split[5] = 0
	end
	local EffectID = tonumber(Split[3])
	local Amplifier = tonumber(Split[5])
	local ApplyEffect = function(OtherPlayer)
		local Seconds = tonumber(Split[4]) * 20
		OtherPlayer:AddEntityEffect(EffectID, Seconds, Amplifier)
		if Split[2] ~= "@a" then
			Player:SendMessageSuccess("Successfully added effect to player \"" .. OtherPlayer:GetName() .. "\" for " .. Split[4] .. " seconds")
		end
	end
	local RemoveEffects = function(OtherPlayer)
		OtherPlayer:ClearEntityEffects()
		if Split[2] ~= "@a" then
			Player:SendMessageSuccess("Successfully removed effects from player \"" .. OtherPlayer:GetName() .. "\"")
		end
	end
	if Split[3] ~= "clear" then
		if Split[3] == nil then
			Player:SendMessageInfo("Usage: " .. Split[1] .. " <player> <effect> [seconds] [amplifier] OR " .. Split[1] .. " <player> clear")
		elseif EffectID == nil or EffectID < 1 or EffectID > 23 then
			Player:SendMessageFailure("Invalid effect ID \"" .. Split[3] .. "\"")
		elseif tonumber(Split[4]) == nil then
			Player:SendMessageFailure("Invalid duration \"" .. Split[4] .. "\"")
		elseif tonumber(Split[4]) < 0 then
			Player:SendMessageFailure("The duration in seconds must be at least 0")
		elseif tonumber(Split[4]) > 1000000 then
			Player:SendMessageFailure("The duration in seconds must be at most 1000000")
		elseif Amplifier == nil then
			Player:SendMessageFailure("Invalid amplification amount \"" .. Split[5] .. "\"")
		elseif Amplifier < 0 then
			Player:SendMessageFailure("The amplification amount must be at least 0")
		elseif Amplifier > 24 and EffectID == 19 or Amplifier > 24 and EffectID == 20 then
			Player:SendMessageFailure("The amplification amount must be at most 24")
		elseif Amplifier > 49 and EffectID == 10 then
			Player:SendMessageFailure("The amplification amount must be at most 49")
		elseif Amplifier > 255 then
			Player:SendMessageFailure("The amplification amount must be at most 255")
		elseif Split[2] == "@a" then
			cRoot:Get():ForEachPlayer(ApplyEffect)
			Player:SendMessageSuccess("Successfully added effect to every player for " .. Split[4] .. " seconds")
		else
			if not cRoot:Get():FindAndDoWithPlayer(Split[2], ApplyEffect) then
				Player:SendMessageFailure("Player \"" .. Split[2] .. "\" not found")
			end
		end
	else
		if Split[2] == "@a" then
			cRoot:Get():ForEachPlayer(RemoveEffects)
			Player:SendMessageSuccess("Successfully removed effects from every player")
		else
			if not cRoot:Get():FindAndDoWithPlayer(Split[2], RemoveEffects) then
				Player:SendMessageFailure("Player \"" .. Split[2] .. "\" not found")
			end
		end
	end
	return true
end
