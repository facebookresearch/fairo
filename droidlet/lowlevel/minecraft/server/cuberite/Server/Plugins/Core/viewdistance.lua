function HandleViewDistanceCommand(a_Split, a_Player)
	-- Check the number of params, show usage if not enough:
	if (#a_Split ~= 2) then
		SendMessage(a_Player, "Usage: /viewdistance <".. cClientHandle.MIN_VIEW_DISTANCE .." - ".. cClientHandle.MAX_VIEW_DISTANCE ..">")
		return true
	end

	-- Check if the param is a number:
	local viewDistance = tonumber(a_Split[2])
	if not(viewDistance) then
		SendMessageFailure(a_Player, "Expected a number parameter")
		return true
	end

	a_Player:GetClientHandle():SetViewDistance(viewDistance)
	SendMessageSuccess(a_Player, "Your view distance is now " .. a_Player:GetClientHandle():GetViewDistance())
	return true
end
