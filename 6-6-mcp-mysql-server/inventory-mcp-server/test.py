from inventory_mcp_server import add_inventory, check_stock, list_inventory


# Add inventory
print(add_inventory("item001", "Laptop", "WarehouseA", 10))

# Check stock
print(check_stock("item001", "WarehouseA"))

# List all inventory
print(list_inventory())
