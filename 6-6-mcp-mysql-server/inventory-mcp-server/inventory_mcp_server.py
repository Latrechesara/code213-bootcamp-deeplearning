from mcp.server.fastmcp import FastMCP
import mysql.connector

# ==============================
# Create the MCP server instance
# ==============================
mcp = FastMCP(name="inventory_mcp")

# ==============================
# MySQL database configuration
# ==============================
db_config = {
    "host": "localhost",
    "user": "root",       
    "password": "rc13@#ch12345CULB!",  
    "database": "aaitech_inventory_dz"  
}

# ==============================
# MCP Tools (Functions) 
# ==============================

@mcp.tool()
def add_inventory(item_id: str, product_name: str, location: str, quantity: int) -> dict:
    """
    Adds a new item to inventory or updates quantity if it already exists.
    Parameters:
        item_id (str): Unique ID of the item
        product_name (str): Name of the product
        location (str): Warehouse location (e.g., Algiers, Oran)
        quantity (int): Number of units to add
    Returns:
        dict: Confirmation message
    """
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO inventory (item_id, product_name, location, quantity) VALUES (%s, %s, %s, %s) "
            "ON DUPLICATE KEY UPDATE quantity = quantity + %s, product_name = VALUES(product_name)",
            (item_id, product_name, location, quantity, quantity)
        )
        conn.commit()
    except mysql.connector.Error as err:
        return {"error": str(err)}
    finally:
        if conn.is_connected():
            conn.close()
    return {"message": f"Added {quantity} units of {product_name} ({item_id}) at {location}"}


@mcp.tool()
def remove_inventory(item_id: str, location: str, quantity: int) -> dict:
    """
    Removes a specified quantity of an item from inventory.
    Parameters:
        item_id (str): Unique ID of the item
        location (str): Warehouse location
        quantity (int): Number of units to remove
    Returns:
        dict: Confirmation message
    """
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE inventory SET quantity = quantity - %s WHERE item_id=%s AND location=%s AND quantity >= %s",
            (quantity, item_id, location, quantity)
        )
        conn.commit()
    except mysql.connector.Error as err:
        return {"error": str(err)}
    finally:
        if conn.is_connected():
            conn.close()
    return {"message": f"Removed {quantity} units of {item_id} from {location}"}


@mcp.tool()
def check_stock(item_id: str, location: str) -> dict:
    """
    Checks the current stock of an item in a specific location.
    Parameters:
        item_id (str): Unique ID of the item
        location (str): Warehouse location
    Returns:
        dict: Stock information (product name and quantity)
    """
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT product_name, quantity FROM inventory WHERE item_id=%s AND location=%s",
            (item_id, location)
        )
        result = cursor.fetchone()
    except mysql.connector.Error as err:
        return {"error": str(err)}
    finally:
        if conn.is_connected():
            conn.close()
    
    if result:
        return {
            "item_id": item_id,
            "location": location,
            "product_name": result[0],
            "quantity": result[1]
        }
    else:
        return {
            "item_id": item_id,
            "location": location,
            "product_name": None,
            "quantity": 0
        }


@mcp.tool()
def list_inventory() -> list:
    """
    Lists all items in the inventory across all locations.
    Returns:
        list: List of dictionaries containing inventory details
    """
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT item_id, product_name, location, quantity FROM inventory")
        rows = cursor.fetchall()
    except mysql.connector.Error as err:
        return {"error": str(err)}
    finally:
        if conn.is_connected():
            conn.close()
    
    return rows


# ==============================
# Start the MCP server
# ==============================
if __name__ == "__main__":
    print("Inventory MCP server started and waiting for clients...")
    mcp.run()

