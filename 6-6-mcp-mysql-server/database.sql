CREATE DATABASE IF NOT EXISTS aaitech_inventory_dz;
USE aaitech_inventory;

CREATE TABLE IF NOT EXISTS inventory (
    item_id VARCHAR(50),
    product_name VARCHAR(100),
    location VARCHAR(50),
    quantity INT DEFAULT 0,
    PRIMARY KEY (item_id, location)
);

INSERT INTO inventory (item_id, product_name, location, quantity) VALUES
('LAP-001', 'Dell Inspiron Laptop', 'Algiers', 25),
('LAP-002', 'HP Pavilion Laptop', 'Oran', 15),
('LAP-003', 'Lenovo ThinkPad Laptop', 'Algiers', 10),
('LAP-004', 'Apple MacBook Air', 'Constantine', 8),
('LAP-005', 'Asus VivoBook', 'Oran', 12),
('LAP-006', 'Acer Aspire 7', 'Annaba', 14),

('MOB-001', 'iPhone 14', 'Algiers', 40),
('MOB-002', 'Samsung Galaxy S23', 'Oran', 35),
('MOB-003', 'Xiaomi Redmi Note 12', 'Constantine', 20),
('MOB-004', 'Huawei P50', 'Algiers', 18),
('MOB-005', 'Infinix Note 30', 'Oran', 22),
('MOB-006', 'Realme 12 Pro', 'Annaba', 16),

('TAB-001', 'Samsung Galaxy Tab S8', 'Constantine', 14),
('TAB-002', 'Apple iPad Air', 'Algiers', 17),
('TAB-003', 'Lenovo Tab M10', 'Oran', 9),
('TAB-004', 'Huawei MatePad', 'Constantine', 11),
('TAB-005', 'Alcatel 3T Tablet', 'Algiers', 13),
('TAB-006', 'iBall Slide', 'Annaba', 8),

('ACC-001', 'Logitech Mouse', 'Oran', 50),
('ACC-002', 'Dell Keyboard', 'Constantine', 45),
('ACC-003', 'HP USB-C Dock', 'Algiers', 60),
('ACC-004', 'Samsung 25W Charger', 'Oran', 30),
('ACC-005', 'Apple AirPods', 'Constantine', 55),
('ACC-006', 'Xiaomi Earbuds', 'Annaba', 20);
