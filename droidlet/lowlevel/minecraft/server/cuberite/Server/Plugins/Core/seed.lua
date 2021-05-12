function HandleSeedCommand( Split, Player )
	if not(Split[2]) then
		Player:SendMessageInfo("Seed: " .. Player:GetWorld():GetSeed())
	else
		local World = cRoot:Get():GetWorld(Split[2])
		if not(World) then
			Player:SendMessageInfo("There is no world \"" .. Split[2] .. "\"")
		else
			Player:SendMessageInfo("Seed: " .. World:GetSeed())
		end
	end
	return true
end





function HandleConsoleSeed( Split )
	if not(Split[2]) then
		return true, "Seed: " .. cRoot:Get():GetDefaultWorld():GetSeed()
	else
		local World = cRoot:Get():GetWorld(Split[2])
		if  not(World) then
			return true, "There is no world \"" .. Split[2] .. "\""
		else
			return true, "Seed: " .. World:GetSeed()
		end
	end
end
