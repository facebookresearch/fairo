-- Based on the enumeration class by Waldemar Celes found in enumerate.lua
-- Following is the original notice from that file.

-- This code is free software; you can redistribute it and/or modify it.
-- The software provided hereunder is on an "as is" basis, and
-- the author has no obligation to provide maintenance, support, updates,
-- enhancements, or modifications.


-- ScopedEnum class
-- Represents enumeration
-- The following fields are stored:
--    {i} = list of constant names
classScopedEnum = {
}
classScopedEnum.__index = classScopedEnum
setmetatable(classScopedEnum,classFeature)

-- register scopedenum
function classScopedEnum:register (pre)
	if not self:check_public_access() then
		return
	end
 pre = pre or ''
 local nspace = getnamespace(classContainer.curr)
 local i=1
 output(pre..'tolua_module(tolua_S,"'..self.name..'",0);')
 output(pre..'tolua_beginmodule(tolua_S,"'..self.name..'");')
 while self[i] do
 	if self.lnames[i] and self.lnames[i] ~= "" then
		output(pre..' tolua_constant(tolua_S,"'..self.lnames[i]..'",static_cast<lua_Number>('..nspace..self.name.."::"..self[i]..'));')
	end
  i = i+1
 end
 output(pre..'tolua_endmodule(tolua_S);')
end

-- Print method
function classScopedEnum:print (ident,close)
 print(ident.."ScopedEnum{")
 print(ident.." name = "..self.name)
 local i=1
 while self[i] do
  print(ident.." '"..self[i].."'("..self.lnames[i].."),")
  i = i+1
 end
 print(ident.."}"..close)
end

function emitenumprototype(type)
 output("int tolua_is" .. string.gsub(type,"::","_") .. " (lua_State* L, int lo, int def, tolua_Error* err);")
end

_global_output_enums = {}

-- write support code
function classScopedEnum:supcode ()
	if _global_output_enums[self.name] == nil then
		_global_output_enums[self.name] = 1
		output("int tolua_is" .. string.gsub(self.name,"::","_") .. " (lua_State* L, int lo, int def, tolua_Error* err)")
		output("{")
		output("\tif (!tolua_isnumber(L,lo,def,err)) return 0;")
		output("\tlua_Number val = tolua_tonumber(L,lo,def);")
		output("\tif (val >= " .. self.min .. ".0 && val <= " ..self.max .. ".0) return 1;")
		output("\terr->index = lo;")
		output("\terr->array = 0;")
		output("\terr->type = \"" .. self.name .. "\";")
		output("\treturn 0;")
		output("}")
	end
end

-- Internal constructor
function _ScopedEnum (t,varname)
 setmetatable(t,classScopedEnum)
 append(t)
 appendenum(t)
	 local parent = classContainer.curr
	 if parent then
		t.access = parent.curr_member_access
		t.global_access = t:check_public_access()
	 end
	return t
end

-- Constructor
-- Expects a string representing the enumerate body
function ScopedEnum (n,b,varname,typed)
	b = string.gsub(b, ",[%s\n]*}", "\n}") -- eliminate last ','
	local t = split(strsub(b,2,-2),',') -- eliminate braces
	local i = 1
	local e = {n=0}
	local value = 0
	local min = 0
	local max = 0
	while t[i] do
		local tt = split(t[i],'=')  -- discard initial value
		e.n = e.n + 1
		e[e.n] = tt[1]
		tt[2] = tonumber(tt[2])
		if tt[2] == nil then
			tt[2] = value
		end 
  		value = tt[2] + 1 -- advance the selected value
		if tt[2] > max then
			max = tt[2]
		end
		if tt[2] < min then
			min = tt[2]
		end
		i = i+1
	end
	-- set lua names
	i  = 1
	e.lnames = {}
	local ns = getcurrnamespace()
	while e[i] do
		local t = split(e[i],'@')
		e[i] = t[1]
		if not t[2] then
			t[2] = applyrenaming(t[1])
		end
		e.lnames[i] = t[2] or t[1]
		local fullname = ns.."::"..n.."::"..e[i]
		_global_enums[ fullname ] = (fullname)
		i = i+1
	end
	e.name = n
	e.min = min
	e.max = max
	if n ~= "" then
		_enums[n] = true
		if typed and typed ~= "" then
			Typedef(typed.." "..n)
		else
			Typedef("int "..n)
		end
	end
	return _ScopedEnum(e, varname)
end

