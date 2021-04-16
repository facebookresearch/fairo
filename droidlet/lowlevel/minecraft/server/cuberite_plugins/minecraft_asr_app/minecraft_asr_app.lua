-- Copyright (c) Facebook, Inc. and its affiliates.

PLUGIN = nil
g_Server = nil
g_Players = {} --table that has current joined players (not including bot)
g_numPlayers = 0 --num of players (not including bot) server starting/closing is dependent on this variable
g_Port = 2556
g_WebPort = 8000
g_IPaddr = "127.0.0.1"
g_Bot = nil --set when bot joins the game so can send message to bot

local ProcessMessage =
-- Handles the incoming connection
{
	OnConnected = function (a_Link)
		-- This will not be called for a server connection, ever
		assert(false, "minecraft_asr_app: Unexpected Connect callback call")
	end,

	OnError = function (a_Link, a_ErrorCode, a_ErrorMsg)
		-- Log the error to console:
		local RemoteName = "'" .. a_Link:GetRemoteIP() .. ":" .. a_Link:GetRemotePort() .. "'"
		LOG("speech_app: an error has occurred while talking to " .. RemoteName .. ": " .. a_ErrorCode .. " (" .. a_ErrorMsg .. ")")
	end,

	OnReceivedData = function (a_Link, a_Data)
		-- parse the data received and send it to the bot and player accordingly
		LOG("minecraft_asr_app: received data: " .. a_Data)
		local returnMessage = ""
		i,j = string.find(a_Data,"::")
		if (i ~= nil) then
			u = string.sub(a_Data,1,i-1)
			m = string.sub(a_Data,j+1)
			player = g_Players[u]
			if (player == nil) then
				LOG("minecraft_asr_app: error, player not found")
				returnMessage = "Player with the username: "..u.." not found"
			else
				--format message to include username
				local message = "<" .. player:GetName() .. "> " .. m
				-- send the message to all players
				local world = player:GetWorld()
				local callbackSendMessage = function (player)
					player:SendMessage(message)
				end
				world:ForEachPlayer(callbackSendMessage)
				returnMessage = "success: message sent"
			end
		else
			returnMessage = "error: data not formatted correctly"
		end
		a_Link:Send(returnMessage)
	end,

	OnRemoteClosed = function (a_Link)
		-- Log the event into the console:
		local RemoteName = "'" .. a_Link:GetRemoteIP() .. ":" .. a_Link:GetRemotePort() .. "'"
		LOG("minecraft_asr_app: connection to '" .. RemoteName .. "' closed")
	end,
}

local ListenCallbacks =
-- Define the clalbacks used by the server
{
	OnAccepted = function (a_Link)
		-- No processing needed, just log that this happened:
		LOG("minecraft_asr_app: accepted connection")
	end,

	OnError = function (a_ErrorCode, a_ErrorMsg)
		-- An error has occured while listening for incoming connections, log it:
		LOG("minecraft_asr_app: cannot listen, error " .. a_ErrorCode .. " (" .. a_ErrorMsg .. ")")
	end,

	OnIncomingConnection = function (a_RemoteIP, a_RemotePort, a_LocalPort)
		-- A new connection is being accepted, process the data sent
		LOG("minecraft_asr_app: incoming Connection" .. a_RemoteIP)
		return ProcessMessage
	end,
}

function Initialize(Plugin)
  Plugin:SetName("ASR App")

  PLUGIN = Plugin
  cPluginManager.AddHook(cPluginManager.HOOK_PLAYER_DESTROYED, OnPlayerDestroyed)
  cPluginManager.AddHook(cPluginManager.HOOK_PLAYER_JOINED, OnPlayerJoined)
  LOG("Loaded plugin: ASR App")

	StartWebserver()

  return true
end

function isBot(username) 
-- isBot cehcks if the username string starts with "bot."
  if (string.sub(username,1,4) == "bot.") then
		return true
	else
		return false
	end
end

function StartWebserver()
--get the path of the python scirpt and execute it with the info
  local info = g_IPaddr..","..g_Port
	local path = (debug.getinfo(2, "S").source:sub(2)):match("(.*/)")
	os.execute("python " .. path .. "/qr_web.py -port " .. g_WebPort .. " -info " .. info .. " &")
end

function updateQRCode(username)
--get the path of python script and execute it just to update the qrcode image
	local info = g_IPaddr..","..g_Port
	if (username ~= "") then
		info = info .. "," .. username
	end
	local path = (debug.getinfo(2,"S").source:sub(2)):match("(.*/)")
	os.execute("python " .. path .. "/qr_web.py -update -info " .. info)
end

function OnPlayerDestroyed(player)
--updates g_numPlayers count and if there are no more players then close the server
	if isBot(player:GetName()) then
		g_Bot = nil
	else
		g_Players[player:GetName()] = nil
		g_numPlayers = g_numPlayers - 1
		--close the server if there are no more players (not including bot)
		if (g_numPlayers == 0) then
			LOG("minecraft_asr_app: no more players, closing the server")
			g_Server:Close()
			g_Server = nil
		elseif (g_numPlayers == 1) then
			local username = ""
			for key,value in pairs(g_Players) do
				username = key
			end
			if (username ~= nil) then
				updateQRCode(username)
			end
	  end
	end
end

function OnPlayerJoined(player)
--updates g_numPlayers count and opens the server once a player joins
	LOG("minecraft_asr_app: player joined, adding player")
	if isBot(player:GetName()) then
		g_Bot = player --set the bot variable so can send messages to bot
	else
		if (g_Server == nil) then
			g_Server = cNetwork:Listen(g_Port,ListenCallbacks);
	    if not (g_Server:IsListening()) then
	      LOG("minecraft_asr_app: server listening error")
	      return;
	    end
	    LOG("minecraft_asr_app: listening on port: " .. g_Port)
		end
		g_Players[player:GetName()] = player
		g_numPlayers = g_numPlayers + 1
		if g_numPlayers == 1 then
			updateQRCode(player:GetName())
		elseif g_numPlayers == 2 then
			updateQRCode("")
		end		
  end
end
